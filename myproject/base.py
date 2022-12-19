import typing
import abc
import numpy as np
import pandas as pd
from utils.utils import (
    load_test_data,
    filter_city_data,
)


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

S = 52


def process_features(data):
    """Preprocess base dataframe features"""
    data["week_start_date"] = pd.to_datetime(data.week_start_date)
    data["month"] = data.week_start_date.dt.month.astype(float)
    return data.ffill(axis=0)


class BaseDengueModel(abc.ABC):
    """
    Base Model for Dengue AI challenge

    Args:
        data: training dataframe with features
        city_name: name of the specific city the mdoel is trained for
        test_period: number of days to isolate from model training process
                     bbbto simulate performance metrics
    """

    def __init__(
        self,
        data: pd.DataFrame,
        city_name: str,
        test_period: int = 90,
    ) -> None:
        self.city_name = city_name
        self.data = filter_city_data(data, city_name)
        self.test_period = test_period
        self.train_data = process_features(self.data)

    def split_train_test(self, data):
        """Split train/test dataset for model performance measurement"""
        train_data = data.iloc[: -self.test_period]
        test_data = data.iloc[-self.test_period :]
        return train_data, test_data

    def generate_submission_file(
        self, filepath: str = "data/", save_to_separate_csv=True
    ):
        """Run prediction on test dataset and formulate submission file"""
        # load test_data
        test_data = load_test_data(filepath)
        test_data = filter_city_data(test_data, self.city_name)

        test_data = test_data.reset_index()
        test_data = test_data.fillna(0)
        # generate predictions
        forecast = self.predict(test_data)
        forecast_df = pd.DataFrame()
        forecast_df[["city", "year", "weekofyear"]] = test_data[
            ["city", "year", "weekofyear"]
        ]
        forecast_df["forecast"] = forecast.values

        forecast_df = forecast_df.fillna(0)
        forecast_df["forecast"] = forecast_df["forecast"].apply(
            lambda x: max(0, round(x))
        )
        # format files and save
        if save_to_separate_csv:
            forecast_df.to_csv(f"{self.city_name}_forecast.csv")
            forecast_df["forecast"].plot()

        return forecast_df

    def get_validation_data_metrics(self, test_data: pd.DataFrame, plot=True):
        """Generate metrics for validation data set"""
        test_data.reset_index(inplace=True)
        forecast = self.predict(test_data)
        result_dataframe = pd.DataFrame()
        result_dataframe["total_cases"] = forecast
        result_dataframe.reset_index(inplace=True)
        result_dataframe["forecast"] = test_data["total_cases"]
        print(
            f"test dataset mae: {np.nanmean(abs(result_dataframe.total_cases - result_dataframe.forecast))}"
        )
        if plot:
            result_dataframe[["total_cases", "forecast"]].plot(figsize=(12, 8))

    @abc.abstractmethod
    def train_models(
        self, get_model_summary: bool = True, run_param_search: bool = False
    ) -> None:
        """Model specific train method"""

    @abc.abstractmethod
    def predict(self, new_data: pd.DataFrame) -> typing.Union[pd.DataFrame, pd.Series]:
        """Model specific predict method"""
