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


def map_breast_surg_type(code):
    '''
    this map method is based on the Surgery_Codes_Breast_2021.pdf from https://seer.cancer.gov/archive/manuals/2021/AppendixC/Surgery_Codes_Breast_2021.pdf
    TODO:unit test in the future
    '''
    if is_integer(code) is False:
        raise ValueError(form_error_msg("Invalid parameter code."))
    elif code == 0:
        return "None"
    elif code == 19:
        return "Local tumor destruction"
    elif code == 20 or code == 21 or code == 22 or code == 23 or code == 24:
        return "Partial mastectomy"
    elif code == 30:
        return "Subcutaneous mastectomy"
    elif code == 40 or code == 41 or code == 43 or code == 44 or code == 45 or code == 46 or code == 42 or code == 47 or code == 48 or code == 49 or code == 75:
        return "Total (simple) mastectomy"
    elif code == 76:
        return "Bilateral mastectomy"
    elif code == 50 or code == 51 or code == 53 or code == 54 or code == 55 or code == 56 or code == 52 or code == 57 or code == 58 or code == 59 or code == 63:
        return "Modified radical mastectomy"
    elif code == 60 or code == 61 or code == 64 or code == 65 or code == 66 or code == 67 or code == 62 or code == 68 or code == 69 or code == 73 or code == 74:
        return "Radical mastectomy"
    elif code == 70 or code == 71 or code == 72:
        return "Extended radical mastectomy"
    elif code == 80:
        return "Mastectomy"
    else:
        raise ValueError(form_error_msg("Invalid parameter code."))
