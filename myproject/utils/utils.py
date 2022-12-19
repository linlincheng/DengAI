import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def load_data(datapath: str = "data/") -> pd.DataFrame:
    """
    Load train data from local directory
    Data obtained from:
    https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/
    """
    train_data = pd.read_csv(datapath + "dengue_features_train.csv")
    train_label = pd.read_csv(datapath + "dengue_labels_train.csv")
    dataset = train_data.merge(
        train_label, how="inner", on=["city", "year", "weekofyear"]
    )
    return dataset


def load_test_data(datapath: str = "data/") -> pd.DataFrame:
    """
    Load test data for predictions from local directory:
    Data obtained from:
    https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/
    """
    test_data = pd.read_csv(datapath + "dengue_features_test.csv", parse_dates=[3])
    return test_data


def filter_city_data(data: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """
    Filter helper to keep only data from specified city_name
    """
    return data[data.city == city_name]


def target_data_stationarity_diagnosis(target: pd.DataFrame) -> None:
    """
    Diagnositc function to help understand series stationality and differencing requirement
    """
    ad_fuller_result = adfuller(target)
    ad_full_p_value = ad_fuller_result[1]
    print(f"ADF Statistic: {ad_fuller_result[0]}")
    print(f"p-value: {ad_full_p_value}")
    if ad_full_p_value <= 0.05:
        print("p-value less than 0.05, series likely stationary")
    else:
        print("p-value greater than 0.05, series likely nonstationary, please check")


class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
