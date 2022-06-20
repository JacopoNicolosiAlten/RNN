import os
import pickle as pkl
from re import A
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import datetime as dt
from sklearn.preprocessing import StandardScaler

import_data_path = r"C:\Users\jnicolosi\Desktop\tesi\data"
export_data_path = r"C:\Users\jnicolosi\Desktop\tesi\codes\RNN\ts version\data"
forecast_target = dt.timedelta(minutes=40)

seq_len = 12
limit = 2
min_col_rate = .7

def help():
    print(f"The main function reads 'dataset.csv' form {import_data_path}",
            "and stores X_train, y_train, X_test, y_test and features as numpy",
            f"arrays with pickle in {export_data_path}.")

def scalar_treatment(df):
    '''
    standardize scalar features
    '''
    features_to_scale = df.columns[~(df.columns.str.endswith('Indicator')\
                                        | df.columns.str.startswith('RC_'))]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

def fill_na(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    '''
    Fill na sequences of every column of df with lenght less or equal than limit.
    The inputed value is the previous available value of the column

    Arguments
    ---
    df : the dataframe to be filled
    limit : the maximum number of consecutive nas that are filled

    Returns
    ---
    the filled dataframe
    '''

    def get_mask(series, limit):
        na = series.isna()
        consecutive_na = na.groupby((~na).cumsum()).transform('sum')
        no_previous = pd.Series(False, index=series.index)
        no_previous.iloc[:consecutive_na[0]] = True
        return ((consecutive_na > limit) | no_previous) | (~na)

    def get_filler(series):
        na = series.isna()
        return series.groupby((~na).cumsum()).transform('sum')
    
    mask = df.apply(lambda series: get_mask(series, limit), axis='index')
    filler = df.apply(get_filler, axis='index')
    return df.where(mask, filler)

def get_tseries(df: pd.DataFrame, seq_len: int) -> tuple:
    '''
    get sliding time series of length seq_len from the dataframe df

    Arguments
    ---
    df : the dataframe from which the time series have to be extracted
    seq_len : the length of the time series

    Returns
    ---
    1D numpy array with length (n_samples - (seq_len - 1)) of the times of the ends of the series

    numpy array with shape (n_samples - (seq_len - 1), seq_len, n_features) that contains the series
    '''
    index = df.index[seq_len - 1:]
    windows = df.rolling(seq_len)
    # skip the first seq_len - 1 windows
    values = [window.values for window in windows][seq_len - 1:]
    X = np.array(values)
    index = np.array(index)
    return (index, X)

def get_na_flags(X: np.ndarray) -> np.ndarray:
    '''
    get a bidimensional boolean array, each row tells if the column of the corresponding series of X contains a NA value
    '''
    return np.isnan(X).any(axis=1)

def get_samples_to_keep(na_flags: np.ndarray, min_col_rate: float) -> np.ndarray:
    '''
    Returns
    ---
    1D boolean np.ndarray that tells if the corresponding series has to be dropped
    '''
    return na_flags.mean(axis=1) < 1 - min_col_rate

def get_Xy(dataset: pd.DataFrame, limit: int, seq_len: int, min_col_rate: float) -> tuple:
    y = pd.DataFrame({'target': dataset.loc[dataset['Quantità MVM'] > 10,'%Error_High']})
    y['target'] = (y['target'] > 20).astype('bool')
    y.index = y.index - forecast_target
    scalar_treatment(dataset)
    df = fill_na(dataset, limit)
    X_index, X = get_tseries(df, seq_len)
    na_flags = get_na_flags(X)
    samples_to_keep = get_samples_to_keep(na_flags, min_col_rate)
    na_flags = np.repeat(na_flags[:, np.newaxis, :], seq_len, axis=1)
    X[na_flags] = np.nan
    X_index = X_index[samples_to_keep]
    X = X[samples_to_keep, :, :]
    index = y.index.intersection(X_index)
    y = y.loc[index]
    X = X[np.isin(X_index, index), :, :]
    return (X, y.values)

def main():
    # load dataset
    dataset = pd.read_csv(os.path.join(import_data_path, 'dataset.csv'))
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset.set_index('Date', inplace=True)

    features = dataset.columns.to_numpy()

    X, y = get_Xy(dataset, limit=limit, seq_len=seq_len, min_col_rate=min_col_rate)

    # train-test split
    test_size_rate = .2
    split_index = round(X.shape[0] * test_size_rate)
    X_train = X[:-split_index, :, :]
    X_test = X[-split_index:, :, :]
    y_train = y[:-split_index, :]
    y_test = y[-split_index:, :]

    # load data
    with open(os.path.join(export_data_path, 'X_train.pickle'), 'wb') as file:
        pkl.dump(X_train, file)
    with open(os.path.join(export_data_path, 'X_test.pickle'), 'wb') as file:
        pkl.dump(X_test, file)
    with open(os.path.join(export_data_path, 'y_train.pickle'), 'wb') as file:
        pkl.dump(y_train, file)
    with open(os.path.join(export_data_path, 'y_test.pickle'), 'wb') as file:
        pkl.dump(y_test, file)
    with open(os.path.join(export_data_path, 'features.pickle'), 'wb') as file:
        pkl.dump(features, file)