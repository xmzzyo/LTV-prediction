'''
We have deleted some data preprocessing details in this file, as it contained some sensitive information, e.g. table and field names of database.
Please use your customized dataset reader here.
'''


import math
import os
import pickle
import re
from datetime import datetime, timedelta

import torch
from scipy.stats import kstest
from sklearn.model_selection import train_test_split

from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

MAX_DAY_TIME = 120
CHURN_INACTIVE_DAY = 15
LOG_DAYS = 30

START_DATE = "2020-03-21"

CAT_FEATS = []
LST_FEATS = ["arpu_list", "avg_sess_time_list", "max_sess_time_list", "action_cnt_list"]


def tabulate_print(data_frame, rows=5, headers="keys", ):
    print(tabulate(data_frame.head(rows), headers=headers, tablefmt='psql'))



def feature_post(features_df, CTG_FEATS):
    rfm = features_df["arpu_list"].apply(get_RFM)
    features_df.loc[:, "recency"] = rfm.apply(lambda x: x[0])
    features_df.loc[:, "frequency"] = rfm.apply(lambda x: x[1])
    features_df.loc[:, "monetary"] = rfm.apply(lambda x: x[2])
    # convert data type
    features_df[CTG_FEATS] = features_df[CTG_FEATS].astype('object')
    features_df.loc[:, "age"] = features_df.loc[:, "age"].astype('int')
    return features_df


def get_RFM(arpu_list, s=0, e=LOG_DAYS):
    arpu_list = arpu_list[s:e]
    days = [s + i for i in range(len(arpu_list)) if arpu_list[i] > 0] + [s]
    r = e - max(days) - 1
    f = len(days) - 1
    m = sum(arpu_list)
    return r, f, m


def convert_rvn_list(rvns):
    rvn_list = [0.0] * MAX_DAY_TIME
    for k, v in rvns.items():
        rvn_list[k] = v
    return rvn_list


def parse_arpu_list(arpu_list):
    ltv_list = dict()
    detail_list = dict()
    is_churn = 1
    if len(arpu_list) == 0:
        return is_churn, ltv_list
    revenue_dict = arpu_list.split(" ")
    for rev in revenue_dict:
        if len(rev) == 0:
            continue
        rev = rev.split(":")
        ltv_list[int(rev[0])] = float(rev[1].split("@")[-1])
    if max(ltv_list.keys()) > MAX_DAY_TIME - CHURN_INACTIVE_DAY:
        is_churn = 0
    return is_churn, convert_rvn_list(ltv_list)


def get_window_RFM(data_list, end_idx=LOG_DAYS, window_size=7):
    rs, fs, ms = [], [], []
    for i in range(int(end_idx / window_size) + 1):
        r, f, m = get_RFM(data_list, s=i * window_size, e=min(end_idx, (i + 1) * window_size))
        rs.append(r)
        fs.append(f)
        ms.append(m)
    return rs, fs, ms


def gnn_feature(df):
    """
    ["arpu_list", "avg_sess_time_list", "max_sess_time_list", "action_cnt_list", "action_type_cnt_list"]
    Args:
        df:

    Returns:

    """
    rfm = df["arpu_list"].apply(get_window_RFM)
    df["arpu_r"] = rfm.apply(lambda x: x[0])
    df["arpu_f"] = rfm.apply(lambda x: x[1])
    df["arpu_m"] = rfm.apply(lambda x: x[2])

    rfm = df["avg_sess_time_list"].apply(get_window_RFM)
    df["avg_sess_r"] = rfm.apply(lambda x: x[0])
    df["avg_sess_f"] = rfm.apply(lambda x: x[1])
    df["avg_sess_m"] = rfm.apply(lambda x: x[2])

    rfm = df["max_sess_time_list"].apply(get_window_RFM)
    df["max_sess_r"] = rfm.apply(lambda x: x[0])
    df["max_sess_f"] = rfm.apply(lambda x: x[1])
    df["max_sess_m"] = rfm.apply(lambda x: x[2])

    rfm = df["action_cnt_list"].apply(get_window_RFM)
    df["act_cnt_r"] = rfm.apply(lambda x: x[0])
    df["act_cnt_f"] = rfm.apply(lambda x: x[1])
    df["act_cnt_m"] = rfm.apply(lambda x: x[2])
    return df


def get_dataset(data_name, do_log=False, future_len=90, limit_num=-1, active_days=-1):
    data_folder = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."), 'data')
    dataset_path = os.path.join(data_folder, f"pickles/online/{data_name}.pickle")
    # dataset_path = os.path.join(data_folder, f"online/{data_name}.pickle")
    if os.path.exists(dataset_path):
        print("load dataset from pickle...")
        ts_x, gnn_x, all_ts = pickle.load(open(dataset_path, "rb"))
        # split2(ts_x, gnn_x, all_ts)
    else:
        df = pickle.load(open(os.path.join(data_folder, f"online/{data_name}.pkl"), "rb"))

        df.loc[:, "arpu_list"] = df["arpu_list"].apply(make_stationary)
        tabulate_print(df)
        df = feature_post(df, CAT_FEATS)
        df = gnn_feature(df)

        all_ts = np.array([*df["arpu_list"].values])

        ts_x = all_ts[:, :LOG_DAYS]

        # 82
        gnn_x = np.array(pd.get_dummies(df.loc[:, CAT_FEATS]), dtype=np.int)
        for feat in ["arpu", "avg_sess", "max_sess", "act_cnt"]:
            gnn_x = np.concatenate([gnn_x, [*df.loc[:, feat + "_r"].values]], axis=1)
            gnn_x = np.concatenate([gnn_x, [*df.loc[:, feat + "_f"].values]], axis=1)
            gnn_x = np.concatenate([gnn_x, [*df.loc[:, feat + "_m"].values]], axis=1)
        # 128
        gnn_x = np.concatenate([gnn_x, [*df.loc[:, "action_type_cnt_list"].values]], axis=1)
        pickle.dump((ts_x, gnn_x, all_ts), open(dataset_path, "wb"))

    if 0 < limit_num < len(ts_x):
        ts_x, gnn_x, all_ts = ts_x[:limit_num], gnn_x[:limit_num], all_ts[:limit_num]

    elif limit_num == -1:
        limit_num = len(ts_x)

    y = np.sum(all_ts[:, :LOG_DAYS + future_len], axis=1)

    print(f"statistic ltv info for {data_name}, future len is {future_len}")
    print(pd.Series(y).describe())
    print("Consumption freq is :", np.sum(all_ts > 0) / limit_num)
    print("Avg LTV is :", np.sum(all_ts) / limit_num)
    print("Total dataset size is : ", len(ts_x))
    print(f"gnn_x shape: {gnn_x.shape}")

    if do_log:
        return torch.FloatTensor(ts_x), torch.FloatTensor(gnn_x), torch.FloatTensor(np.log1p(y))
    else:
        return torch.FloatTensor(ts_x), torch.FloatTensor(gnn_x), torch.FloatTensor(y)


class TSDataset(Dataset):
    def __init__(self, uid, ts_x, gnn_x, y):
        self.uid = uid
        self.ts_x = ts_x
        self.gnn_x = gnn_x
        self.y = y

    def __getitem__(self, index):
        return self.uid[index], self.ts_x[index], self.gnn_x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def get_dataloaders(data_name, batch_size=2, num_workers=1, test_size=0.4, val_size=0.2, pin_memory=True,
                    future_len=30, limit_num=-1, train_ratio=1.0, active_days=-1):
    ts_x, gnn_x, y = get_dataset(data_name, future_len=future_len, limit_num=limit_num, active_days=active_days)
    uid = torch.arange(0, ts_x.size(0))

    uid_train, uid_test, ts_x_train, ts_x_test, gnn_x_train, gnn_x_test, y_train, y_test = train_test_split(
        uid, ts_x, gnn_x, y, test_size=test_size, random_state=2021)

    train_size = uid_train.size(0)
    uid_train = uid_train[:int(train_ratio * train_size)]
    ts_x_train = ts_x_train[:int(train_ratio * train_size)]
    gnn_x_train = gnn_x_train[:int(train_ratio * train_size)]
    y_train = y_train[:int(train_ratio * train_size)]

    if val_size > 0:
        uid_val, uid_test, ts_x_val, ts_x_test, gnn_x_val, gnn_x_test, y_val, y_test = train_test_split(
            uid_test, ts_x_test, gnn_x_test, y_test, test_size=val_size / test_size, random_state=2021)

    print(
        f"Train ratio is : {train_ratio},\nTrain data size: {len(uid_train)}\nValid data size: {len(uid_val)}\nTest data size: {len(uid_test)}")

    train_dataset = TSDataset(uid_train, ts_x_train, gnn_x_train, y_train)
    test_dataset = TSDataset(uid_test, ts_x_test, gnn_x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  pin_memory=pin_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                 pin_memory=pin_memory)
    val_dataloader = None
    if val_size > 0:
        val_dataset = TSDataset(uid_val, ts_x_val, gnn_x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                    pin_memory=pin_memory)

    return train_dataloader, val_dataloader, test_dataloader


def check_stationary(arpu_df):
    adf = []
    for row in arpu_df.values:
        arpu_list = np.array(row)
        dftest = adfuller(arpu_list, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        adf.append(dfoutput)
    adf = pd.DataFrame(adf)
    adf["percent-1%"] = (adf["Test Statistic"] - adf["Critical Value (1%)"]) / adf["Critical Value (1%)"]
    adf["percent-5%"] = (adf["Test Statistic"] - adf["Critical Value (5%)"]) / adf["Critical Value (5%)"]
    adf["percent-10%"] = (adf["Test Statistic"] - adf["Critical Value (10%)"]) / adf["Critical Value (10%)"]
    tabulate_print(adf, rows=50)
    print(adf["percent-1%"].mean())
    print(adf["percent-5%"].mean())
    print(adf["percent-10%"].mean())

    # result = adfuller(arpu_df.values)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))


def make_stationary(arpu_list):
    # to alleviate MSE loss sensitive
    arpu_list = np.log1p(arpu_list)
    return arpu_list


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def seaborn_hist(data):
    sns.histplot(data=data, bins=30, x="Users'_active_days", y="Normalized_frequency", stat="probability",
                 discrete=True)


if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloaders()
    for uid, ts_x, feat_x, total_y, month_y, month_mask, week_y, week_mask, day_y, day_mask in train_dataloader:
        print(total_y)
        print(month_y)
        print(month_mask)
        print()
        print(week_y)
        print(week_mask)
        print()
        print(day_y)
        print(day_mask)
        print("*******************\n\n\n\n")
