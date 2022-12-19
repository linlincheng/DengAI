import pandas as pd
from prophet_model import ProphetDengModel
from utils.utils import (
    load_data,
    filter_city_data,
)


def execute_model_pipeline(data: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """
    Executing pipeline to:
        1) extract city specific data
        2) train a prophet model
        3) generate a submission file dataframe
    """
    city_data = filter_city_data(data, city_name=city_name)
    # train models
    city_prophet = ProphetDengModel(data=city_data, city_name=city_name, test_period=0)
    city_prophet.train_models(get_model_summary=False, run_param_search=True)
    submission_data = city_prophet.generate_submission_file()
    return submission_data


def run_dengai() -> None:
    """
    Main wrapper function to Load raw data for training and generate final submission file
    """
    # load data
    data = load_data()
    city_list = ["sj", "iq"]
    submission_list = [execute_model_pipeline(data, city) for city in city_list]
    # merge prediction files for submissions
    final_submission_file = pd.concat(submission_list)
    final_submission_file.to_csv("./submission.csv")


if __name__ == "__main__":
    print("Running model training and prediction pipeline for submissions...")
    run_dengai()
