# -*- coding: utf-8 -*-
import random
from itertools import chain

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from sklearn import metrics
from typing import Sequence

from tabulate import tabulate


def cumulative_value(
        y_true: Sequence[float],
        y_pred: Sequence[float]
) -> np.ndarray:
    """Calculates cumulative sum of lifetime values over predicted rank.

    Arguments:
      y_true: true lifetime values.
      y_pred: predicted lifetime values.

    Returns:
      res: cumulative sum of lifetime values over predicted rank.
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
    }).sort_values(
        by='y_pred', ascending=False)

    return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates gini coefficient over gain charts.

    Arguments:
      df: Each column contains one gain chart. First column must be ground truth.

    Returns:
      gini_result: This dataframe has two columns containing raw and normalized
                   gini coefficient.
    """
    raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
    normalized = raw / raw[0]
    return pd.DataFrame({
        'raw': raw,
        'normalized': normalized
    })[['raw', 'normalized']]


def _normalized_rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred)) / y_true.mean()


def _normalized_mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred) / y_true.mean()


def _aggregate_fn(df):
    return pd.Series({
        'label_mean': np.mean(df['y_true']),
        'pred_mean': np.mean(df['y_pred']),
        'normalized_rmse': _normalized_rmse(df['y_true'], df['y_pred']),
        'normalized_mae': _normalized_mae(df['y_true'], df['y_pred']),
    })


def show_gini_fig(y_true, y_pred):
    total_value = np.sum(y_true)
    cumulative_true = np.cumsum(y_true) / total_value
    gain_actual = cumulative_value(
        y_true, y_pred)
    gain_perfect = cumulative_value(
        y_true, y_true)
    gain = pd.DataFrame({
        'ground_truth': gain_perfect,
        'actual_model': gain_actual
    })
    gain['cumulative_customer'] = (np.arange(len(y_true)) + 1.) / len(y_true)
    ax = gain[[
        'cumulative_customer',
        'ground_truth',
        'actual_model',
    ]].plot(
        x='cumulative_customer', figsize=(8, 5), legend=True)

    ax.legend(['Groundtruth', 'Model'], loc='upper left')

    ax.set_xlabel('Cumulative Fraction of Customers')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim((0, 1.))

    ax.set_ylabel('Cumulative Fraction of Total Lifetime Value')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim((0, 1.05))
    ax.set_title('Gain Chart')
    plt.show()

    gini = gini_from_gain(gain.loc[:, ["ground_truth", "actual_model"]])
    # print(gini)
    return gini


# def NMSE(y_true, y_pred):
#     if not isinstance(y_true, pd.Series):
#         y_true = pd.Series(y_true)
#     if not isinstance(y_pred, pd.Series):
#         y_pred = pd.Series(y_pred)
#     nmse = metrics.mean_squared_error(y_pred, y_true) * y_true.size / sum(
#         y_true.apply(lambda x: x ** 2).values)
#     return nmse


# def NMAE(y_true, y_pred):
#     if not isinstance(y_true, pd.Series):
#         y_true = pd.Series(y_true)
#     if not isinstance(y_pred, pd.Series):
#         y_pred = pd.Series(y_pred)
#     nmae = metrics.mean_absolute_error(y_pred, y_true) * y_true.size / sum(
#         y_true.values)
#     return nmae


def LTV_metrics(y_true, y_pred, perfix="val_total_", description="LTV", do_print=False, do_log=False):
    print(f"Validation size: true-{y_true.shape}, pred-{y_pred.shape}")
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if do_log:
        y_true = np.exp(y_true) - 1
        y_pred = np.exp(y_pred) - 1
    y_pred = np.abs(y_pred)
    if "total" not in perfix:
        y_true = y_true.sum(axis=1)
        y_pred = y_pred.sum(axis=1)
    # if len(y_true.shape) == 2:
    #     y_true = np.array(list(chain(*y_true)))
    # if len(y_pred.shape) == 2:
    #     y_pred = np.array(list(chain(*y_pred)))
    metrics_dict = dict()
    metrics_dict[perfix + "MSE"] = metrics.mean_squared_error(y_true, y_pred)
    metrics_dict[perfix + "MAE"] = metrics.mean_absolute_error(y_true, y_pred)
    if do_print:
        metrics_dict_pd = pd.DataFrame(metrics_dict, index=[0])
        print("\n\n" + "-" * 50 + f" {description} prediction result " + "-" * 50)
        print(tabulate(metrics_dict_pd, headers="keys", tablefmt="psql"))
        print("-" * 50 + f" {description} prediction result " + "-" * 50 + "\n\n")
    return metrics_dict


def MSE(output, target, is_seq=False, do_log=False):
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        assert output.shape[0] == len(target)
        if do_log:
            output = torch.exp(output) - 1
            target = torch.exp(target) - 1
        if is_seq:
            output = torch.sum(output, dim=1)
            target = torch.sum(target, dim=1)
        rmse = F.mse_loss(output, target)
    return rmse.item()


def NMSE(output, target, do_log=True):
    """
    Calculates the NMSE (normalized mean square error)
    of the input signal compared to the target signal.
    Initial values can be discarded with discard=n.
    2007, Georg Holzmann
    """
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        if do_log:
            output = torch.exp(output) - 1
            target = torch.exp(target) - 1
        mse = (target - output) ** 2
        nmse = mse.mean() / (target.std() ** 2)
    return nmse.item()


def MAE(output, target, is_seq=False, do_log=False):
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        assert output.shape[0] == len(target)
        if do_log:
            output = torch.exp(output) - 1
            target = torch.exp(target) - 1
        if is_seq:
            output = torch.sum(output, dim=1)
            target = torch.sum(target, dim=1)
        # print("pred: ", output[0].item(), "true: ", target[0].item())
        mae = F.l1_loss(output, target)
    return mae.item()


def NMAE(output, target, do_log=True):
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        if do_log:
            output = torch.exp(output) - 1
            target = torch.exp(target) - 1
        mae = torch.abs(target - output)
        nmae = (mae / target).mean()
    return nmae.item()


def ACC(output, target):
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        # print(f"pred: {output[:10]}")
        # print(f"true: {target[:10]}")
        acc = ((output >= 0.5) == target).sum().item() / len(target)
    return acc


def MSLE(output, target, is_seq=False, do_log=False):
    with torch.no_grad():
        output = output.squeeze()
        target = target.squeeze()
        assert output.shape[0] == len(target)
        if do_log:
            output = torch.exp(output) - 1
            target = torch.exp(target) - 1
        if is_seq:
            output = torch.sum(output, dim=1)
            target = torch.sum(target, dim=1)
        rmsle = MSE(torch.log1p(output), torch.log1p(target), do_log=False)
    return rmsle


if __name__ == "__main__":
    y_true = np.array([[78.5196, 2.6576, 3, 2, 1], [0., 0., 1, 2, 3]])
    y_pred = np.array([[0., 0., 1, 2, 3], [78.5196, 2.6576, 3, 2, 1]])
    # print(np.abs(y_true - y_pred) / (y_true + 1e-6))
    LTV_metrics(y_true, y_pred, perfix="", do_print=True, do_log=True)
    print(MSE(torch.tensor(y_pred), torch.tensor(y_true), is_seq=True))
    print(MSLE(torch.tensor(y_pred), torch.tensor(y_true), is_seq=True))
    print(MAE(torch.tensor(y_pred), torch.tensor(y_true), is_seq=True))
