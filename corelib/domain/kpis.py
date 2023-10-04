# -*- coding: utf-8 -*-
"""KPIs.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Dict,
    List,
    Union,
)

import numpy as np
import pandas as pd


def card_precision_top_k_day(
    df_day: pd.DataFrame, top_k: int = 100
) -> Dict[str, Union[float, List[int]]]:
    """Calculate the card precision top-k per day.

    Args:
        df_day: pd.DataFrame
            Testing data that will be used to calculate the metric.
        top_k: int
            Number of transactions that will be checked.

    Returns:
        Dict[str, Union[float, List[int]]]:
            Card precision top-k.
            Perfect Card precision top-k.
            Detected compromised cards list.
    """
    # Order transactions by decreasing probabilities of frauds
    df_day = (
        df_day.groupby(by="customer_id", as_index=False)
        .agg(scores=("scores", "max"), tx_fraud=("tx_fraud", "max"))
        .sort_values(by="scores", ascending=False)
    )

    # Get the top k most suspicious transactions
    df_day_top_k = df_day.head(top_k)

    results = {
        "card_precision_top_k": df_day_top_k.tx_fraud.mean(),
        "perfect_card_precision_top_k": df_day.tx_fraud.sum() / 100,
        "detected_compromised_cards_list": df_day_top_k[
            df_day_top_k.tx_fraud == 1
        ].customer_id.values.tolist(),
    }

    # % of real frauds out of the k most suspicious
    return results


def card_precision_top_k(
    test_data: pd.DataFrame,
    top_k: int = 100,
    remove_detected_compromised_cards: bool = True,
) -> Dict[str, float]:
    """Calculate the average card precision top-k metric per day.

    Args:
        test_data: pd.DataFrame
            Testing data that will be used to calculate the metric.
        top_k: int
            Number of transactions that will be checked.
        remove_detected_compromised_cards: bool
            If True, previously compromised cards will be removed.

    Returns:
        Tuple[float, float]:
            average card precision top-k.
            perfect average card precision top-k.
    """
    start_date = test_data.tx_datetime.min().floor(freq="D")
    end_date = test_data.tx_datetime.max().ceil(freq="D")
    num_days = (end_date - start_date).days

    list_detected_compromised_cards = []

    precision_top_k_per_day_list = []
    perfect_precision_top_k_per_day_list = []

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
            & ~(test_data.customer_id.isin(list_detected_compromised_cards))
        ]

        results = card_precision_top_k_day(df_day=df_day, top_k=top_k)

        precision_top_k_per_day_list.append(
            results.get("card_precision_top_k")
        )
        perfect_precision_top_k_per_day_list.append(
            results.get("perfect_card_precision_top_k")
        )

        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(
                results.get("detected_compromised_cards_list")
            )

    # Compute the mean
    mean_precision_top_k = np.round(
        np.array(precision_top_k_per_day_list).mean(), 3
    )

    perfect_precision_top_k = np.round(
        np.array(perfect_precision_top_k_per_day_list).mean(), 3
    )

    return mean_precision_top_k, perfect_precision_top_k
