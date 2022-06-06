import inspect
import os.path

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


def read_from_file(filepath):
    if os.path.isfile(filepath) is False:
        raise ValueError(form_error_msg("Invalid parameter filepath."))
    if filepath.endswith(".csv") is False:
        raise ValueError(form_error_msg("Invalid file extension."))
    return pd.read_csv(filepath)


def split_data(data, fraction):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if fraction <= 0 or fraction >= 1:
        raise ValueError(form_error_msg("Invalid parameter test_proportion."))
    return data.sample(frac=fraction)


def remove_data(data, indexs):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    data.drop(indexs)


def get_function_name():
    currentframe = inspect.currentframe()
    return inspect.getframeinfo(currentframe).function


def form_error_msg(error_msg):
    return get_function_name() + ":" + error_msg


def get_data_frame_row_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.shape[0]


def get_data_frame_col_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if len(data_frame) == 1:
        return 1
    return data_frame.shape[1]


def get_data_frame_col_names(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.columns


def filter_col_data(data, cols_array):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols_array) is False:
        raise ValueError(form_error_msg("Invalid parameter cols_array."))
    return data.loc[:, cols_array]


def is_data_frame(data):
    return isinstance(data, pd.DataFrame)


def is_series(data):
    return isinstance(data, pd.Series)


def is_integer(var):
    return isinstance(var, int)


def is_valid_list(array):
    return isinstance(array, list) and array


def one_hot_encode_cols(data, cols):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols) is False:
        raise ValueError(form_error_msg("Invalid parameter cols."))
    transformer = make_column_transformer((OneHotEncoder(), cols), remainder="passthrough")
    return transformer.fit_transform(data)


def print_series_info(series):
    if is_series(series) is False:
        raise ValueError(form_error_msg("Invalid parameter series."))
    print(sorted(series.unique()))


def print_data_frame_info(df):
    if is_data_frame(df) is False:
        raise ValueError(form_error_msg("Invalid parameter df."))
    for col in df:
        print("----------" + col + "----------")
        print_series_info(df[col])
