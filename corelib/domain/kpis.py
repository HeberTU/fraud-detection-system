# -*- coding: utf-8 -*-
"""KPIs.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import numpy as np
import pandas as pd


def precision_top_k_day(df_day: pd.DataFrame, top_k: int = 100) -> float:
    """Calculate the precision top-k per day.

    Args:
        df_day: pd.DataFrame
            Testing data that will be used to calculate the metric.
        top_k: int
            Number of transactions that will be checked.

    Returns:
        float:
            precision top-k.
    """
    # Order transactions by decreasing probabilities of frauds
    df_day = df_day.sort_values(by="scores", ascending=False)

    # Get the top k most suspicious transactions
    df_day_top_k = df_day.head(top_k)

    # % of real frauds out of the k most suspicious
    return df_day_top_k.tx_fraud.mean()


def precision_top_k(test_data: pd.DataFrame, top_k: int = 100) -> float:
    """Calculate the average precision top-k metric per day.

    Args:
        test_data: pd.DataFrame
            Testing data that will be used to calculate the metric.
        top_k: int
            Number of transactions that will be checked.

    Returns:
        float:
            average precision top-k.
    """
    start_date = test_data.tx_datetime.min().floor(freq="D")
    end_date = test_data.tx_datetime.max().ceil(freq="D")
    num_days = (end_date - start_date).days

    precision_top_k_per_day_list = []

    # calculate the precision top-k metric per day.
    for i in range(num_days):

        df_day = test_data[
            (
                test_data.tx_datetime
                <= start_date + pd.Timedelta(value=i + 1, unit="days")
            )
            & (
                test_data.tx_datetime
                > start_date + pd.Timedelta(value=i, unit="days")
            )
        ]

        precision_top_k_value = precision_top_k_day(df_day=df_day, top_k=top_k)

        precision_top_k_per_day_list.append(precision_top_k_value)

    # Compute the mean
    mean_precision_top_k = np.round(
        np.array(precision_top_k_per_day_list).mean(), 3
    )

    return mean_precision_top_k
