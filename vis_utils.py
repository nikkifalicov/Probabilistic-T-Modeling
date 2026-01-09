import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import PolynomialFeatures

# Allow user to specify location of csv files with DATA_DIR env var
# but by default, use folder structure distributed with source code
src_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR',
                          os.path.join(os.path.dirname(src_dir), 'traffic_data'))


def load_dataset(seed=123, val_size=0, data_dir=DATA_DIR):
    # Load and unpack training and test data
    train_csv_fpath = os.path.join(data_dir, 'jan_through_may_data_scaled.csv')
    test_csv_fpath = os.path.join(data_dir, 'june_data_scaled.csv')
    if not os.path.exists(train_csv_fpath):
        raise FileNotFoundError("Please set DATA_DIR. Cannot find CSV files at path: ",
                                train_csv_fpath)

    train_df = pd.read_csv(train_csv_fpath)
    test_df = pd.read_csv(test_csv_fpath)

    feature_cols = [
        'from_stop_departure_sec',
        'days_since_jan1',
        'day_of_week',
        'is_holiday'
    ]
    # x_train_ND = train_df['x'].values[:, np.newaxis]
    x_train_ND = train_df[feature_cols].values
    t_train_N = train_df['travel_time_sec'].values

    random_state = np.random.RandomState(int(seed))
    shuffle_ids = random_state.permutation(t_train_N.size)
    x_train_ND = x_train_ND[shuffle_ids]
    t_train_N = t_train_N[shuffle_ids]

    x_test_ND = test_df[feature_cols].values
    t_test_N = test_df['travel_time_sec'].values

    if val_size == 0:
        return x_train_ND, t_train_N, x_test_ND, t_test_N
    else:
        assert val_size > 0
        V = int(val_size)
        x_val_VD, t_val_V = x_train_ND[-V:], t_train_N[-V:]
        x_train_ND, t_train_N = x_train_ND[:-V], t_train_N[:-V]
        return x_train_ND, t_train_N, x_val_VD, t_val_V


def load_mixed_dataset(seed=123, val_size=0, data_dir=DATA_DIR):
    # Load and unpack training and test data
    train_csv_fpath = os.path.join(data_dir, 'jan_through_may_data_scaled.csv')
    test_csv_fpath = os.path.join(data_dir, 'june_data_scaled.csv')
    if not os.path.exists(train_csv_fpath):
        raise FileNotFoundError("Please set DATA_DIR. Cannot find CSV files at path: ",
                                train_csv_fpath)

    train_df = pd.read_csv(train_csv_fpath)
    test_df = pd.read_csv(test_csv_fpath)

    mixed_df = pd.concat([train_df, test_df], ignore_index=True)

    feature_cols = [
        'from_stop_departure_sec',
        'days_since_jan1',
        'day_of_week',
        'is_holiday'
    ]
    # x_train_ND = train_df['x'].values[:, np.newaxis]
    x_ND = mixed_df[feature_cols].values
    t_N = mixed_df['travel_time_sec'].values

    random_state = np.random.RandomState(int(seed))
    shuffle_ids = random_state.permutation(t_N.size)
    x_ND = x_ND[shuffle_ids]
    t_N = t_N[shuffle_ids]

    partition_size = int(.83 * len(mixed_df))

    x_train_ND = x_ND[:partition_size, :]
    t_train_N = t_N[:partition_size]

    x_test_ND = x_ND[partition_size:, :]
    t_test_N = t_N[partition_size:]

    if val_size == 0:
        return x_train_ND, t_train_N, x_test_ND, t_test_N
    else:
        assert val_size > 0
        V = int(val_size)
        x_val_VD, t_val_V = x_train_ND[-V:], t_train_N[-V:]
        x_train_ND, t_train_N = x_train_ND[:-V], t_train_N[:-V]
        return x_train_ND, t_train_N, x_val_VD, t_val_V
