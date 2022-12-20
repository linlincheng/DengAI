import typing
from itertools import product
import pandas as pd
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.utils import (
    target_data_stationarity_diagnosis,
)
from base import BaseDengueModel


TARGET = "total_cases"
EXOGENOUS_VARS = [
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c",
    "station_diur_temp_rng_c",
    "station_max_temp_c",
    "station_min_temp_c",
    "station_precip_mm",
]

# set seasonality param
S = 4


def set_sarimax_param_grid() -> typing.List:
    """SARIMAX parameter grid combination options"""
    p = range(0, 3, 1)
    d = range(1, 2, 1)
    q = range(0, 3, 1)
    P = range(0, 3, 1)
    D = range(0, 3, 1)
    Q = range(0, 3, 1)
    parameters = product(p, d, q, P, D, Q)
    return list(parameters)


def optimize_SARIMAX(
    target: typing.Union[pd.Series, list],
    exog: typing.Union[pd.Series, list],
    order_list: list,
) -> pd.DataFrame:
    """SARIMAX grid search helper function to loop through parameter combinations"""
    results = []

    for order in tqdm_notebook(order_list):
        try:
            print(order)
            model = SARIMAX(
                target,
                exog,
                order=(order[0], order[1], order[2]),
                seasonal_order=(order[3], order[4], order[5], S),
                simple_differencing=False,
            ).fit(disp=False)
            results.append([order, model.aic])
        except ValueError:
            continue

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,d,q,P,D,Q)", "AIC"]  # type: ignore

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


class SarimaxDengModel(BaseDengueModel):
    """
    SARIMAX model implemenation for DENGAI prediction challenge
        Args:
        data: training dataframe with features
        city_name: name of the specific city the mdoel is trained for
        test_period: number of days to isolate from model training process
                     bbbto simulate performance metrics
    """

    name = "SARIMAX"
    version = "0.0.1"

    def __init__(self, data, city_name, test_period):
        super().__init__(data=data, city_name=city_name, test_period=test_period)
        self.model_fit = None

    def train_models(
        self, get_model_summary: bool = True, run_param_search: bool = False
    ) -> None:
        """
        Train Sarimax model on the imported dataframe.

        Args:
            get_model_summary: whether to print out model metric summary statistics, default to True.
            run_param_search: whether to run parameter grid search for SARIMAX, default to False.
        """
        train_data = self.train_data
        if self.test_period > 0:
            train_data, test_data = self.split_train_test(self.train_data)

        target_data_stationarity_diagnosis(train_data[TARGET])

        if run_param_search:
            # run parameter grid search
            # may take a few hours, do not recommend running on the fly
            parameters_list = set_sarimax_param_grid()
            result_df = optimize_SARIMAX(
                train_data[TARGET], train_data[EXOGENOUS_VARS], parameters_list
            )
            params = tuple(result_df.iloc[0].values)[0]
            print(params)
        else:
            # default parameters
            params = (2, 0, 3, 1, 0, 2)

        model = self.set_model_pipeline(train_data, params=params)
        self.model_fit = model.fit(disp=False)
        if get_model_summary:
            self.model_fit.summary()
            if self.test_period > 0:
                self.get_validation_data_metrics(test_data, plot=True)

    def set_model_pipeline(self, train_data: pd.DataFrame, params: tuple):
        """Setting up Sarimax model speficic pipline"""
        model = SARIMAX(
            train_data[TARGET],
            train_data[EXOGENOUS_VARS],
            order=(params[0], params[1], params[2]),
            seasonal_order=(params[3], params[4], params[5], S),
            simple_differencing=False,
        )
        return model

    def predict(self, new_data: pd.DataFrame) -> typing.Union[pd.DataFrame, pd.Series]:
        """Predicting cases for new data"""
        steps = len(new_data)
        forecast = self.model_fit.get_forecast(
            steps=steps, exog=new_data[EXOGENOUS_VARS]
        )
        return forecast.predicted_mean
