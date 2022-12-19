import copy
import numpy as np

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from base import BaseDengueModel
from utils.utils import target_data_stationarity_diagnosis, suppress_stdout_stderr

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
INDEX = "week_start_date"


def set_prophet_param_grid():
    """Setting up Prophet parameter search grid"""
    grid_params = {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.001, 0.01, 0.1],
        "seasonality_prior_scale": [0.01, 0.1, 1],
        "regressor_prior_scale": [0.01, 0.1, 1],
    }
    return grid_params


def prophet_model_builder(
    df,
    ds,
    target,
    regressors,
    cv_initial,
    cv_horizon,
    n_skip_period,
    growth,
    weekly_seasonality,
    yearly_seasonality,
    changepoint_prior_scale,
    seasonality_prior_scale,
    regressor_prior_scale,
    seasonality_mode,
    verbose,
):
    """
    Set up and process prophet tunable parameters.
    More on documentations here:
    https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.forecasting.fbprophet.Prophet.html
    """
    model = Prophet(
        growth=growth,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
    )

    for regressor in regressors:
        model.add_regressor(name=regressor, prior_scale=regressor_prior_scale)

    # Reformat input
    train = df.loc[:, [ds, target] + regressors]
    train.rename(columns={ds: "ds", target: "y"}, inplace=True)
    # Skip days
    train = train.iloc[n_skip_period:, :]

    # fit
    with suppress_stdout_stderr():
        model.fit(train)
        if cv_horizon and cv_initial:
            df_cv = cross_validation(
                model,
                initial=f"{cv_initial} W",
                horizon=f"{cv_horizon} W",
                parallel="processes",
            )
            pm = performance_metrics(df_cv)
            mae = np.mean(pm["mae"])

    if verbose:
        # print(pm)
        if cv_horizon and cv_initial:
            print(
                f"Cross validation applied with inital {cv_initial} weeks,"
                f"and horizon {cv_horizon} weeks."
            )
            print(f"Average MAE is {mae:.1f}\n")
    return model, mae


def optimize_Prophet(df, params, grid_params, sample=None, get_model_summary=True):
    """
    Wrapper function to run on specified Prophet parameter combinations
        Args:
        df: training dataframe with features
        params: model evaluation and ad hoc parameter set
        grid_params: model specific parameter combination options
        sample: number of grid search combinations to try
        get_model_summary: whether to print out model tuning results, default to True

    """
    tmp_params = copy.deepcopy(params)
    tracking = []
    models = []
    hp_grid = list(ParameterGrid(grid_params))
    cv_metrics = "mae"
    if sample:
        sample = min(len(hp_grid), sample)
        hp_grid = np.random.choice(hp_grid, sample, replace=False)
    print(f"Grid search: searching through {len(hp_grid)} combinations")

    for i, p in enumerate(hp_grid):
        tmp_params.update(**p)
        m, error = prophet_model_builder(df=df, **tmp_params)
        tracking.append(error)
        models.append(m)
        if params["verbose"]:
            print(f"Model {i} result for {p}: {cv_metrics}={error}")

    best_idx = np.argmin(tracking)
    best_model = models[best_idx]
    best_params = tmp_params.update(hp_grid[best_idx])
    best_mape = tracking[best_idx]

    if get_model_summary:
        print(f"Best model is {hp_grid[best_idx]}: {cv_metrics}={best_mape}")
    return best_model, best_params, best_mape


class ProphetDengModel(BaseDengueModel):
    """
    Prophet model implemenation for DENGAI prediction challenge
        Args:
        data: training dataframe with features
        city_name: name of the specific city the mdoel is trained for
        test_period: number of days to isolate from model training process
                     bbbto simulate performance metrics
    """

    name = "PROPHET"
    version = "0.0.1"

    def __init__(self, data, city_name, test_period):
        super().__init__(data=data, city_name=city_name, test_period=test_period)
        self.model_fit = None

    def train_models(
        self, get_model_summary: bool = True, run_param_search: bool = True
    ):
        """
        Train prophet model on the imported dataframe.
        Args:
            get_model_summary: whether to print out model metric summary statistics, default to True.
            run_param_search: whether to run parameter grid search for prophet, default to True.
        """
        train_data = self.train_data
        if self.test_period > 0:
            train_data, _ = self.split_train_test(self.train_data)

        target_data_stationarity_diagnosis(train_data[TARGET])

        if run_param_search:
            grid_params = set_prophet_param_grid()
            params = {
                "ds": INDEX,
                "target": TARGET,
                "regressors": EXOGENOUS_VARS,
                "verbose": False,
                "cv_initial": 5 * 52,
                "cv_horizon": 26,
                "n_skip_period": 0,
                "growth": "linear",
                "weekly_seasonality": False,
                "yearly_seasonality": "auto",
            }
            self.model_fit, _, _ = optimize_Prophet(
                df=train_data,
                params=params,
                grid_params=grid_params,
                sample=20,
                get_model_summary=get_model_summary,
            )
        else:
            raise NotImplementedError("default param not set, please run grid search")

    def predict(self, new_data):
        """Run predictions for new dataset"""
        future_dates = new_data.rename(columns={"week_start_date": "ds"})
        forecast = self.model_fit.predict(future_dates)
        return forecast.yhat
