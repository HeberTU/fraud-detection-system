# -*- coding: utf-8 -*-
"""Aggregated features.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
from typing import List

import pandas as pd

from corelib import utils


class TimeUnits(str, enum.Enum):
    """Available time unites to aggregate."""

    DAYS = "days"


class AggFunc(str, enum.Enum):
    """Available aggregation functions."""

    SUM = "sum"
    COUNT = "count"
    MEAN = "mean"


@utils.cacher
def aggregate_feature(
    transactions_df: pd.DataFrame,
    windows_size_in_days: List[int],
    time_unit: TimeUnits,
    feature_name: str,
    agg_func_list: List[AggFunc],
    datetime_col: str,
    index_name: str,
    grouping_column: str,
) -> pd.DataFrame:
    """Aggregate a feature by a time window and another grouping variable.

    Args:
        transactions_df: pd.DataFrame
            Transactions data frame.
        windows_size_in_days: List[int]
            List of the window sizes to aggregate the feature.
        time_unit: TimeUnits
            Time unit for the window size.
        feature_name: str
            Name of the feature that will be transformed.
        agg_func_list: AggFunc
            List of the Aggregation functions to be applied.
        datetime_col: str
            Name of the timestamp column.
        index_name: str
            Name of the index.
        grouping_column: str
            Outer grouping variable

    Returns:
        pd.DataFrame
            Data frame with the aggregated features.
    """
    transactions_df = transactions_df.groupby(by=grouping_column).apply(
        lambda x: aggregate_feature_by_time_window(
            data=x,
            windows_size_in_days=windows_size_in_days,
            time_unit=time_unit,
            feature_name=feature_name,
            agg_func_list=agg_func_list,
            datetime_col=datetime_col,
            index_name=index_name,
        )
    )

    return transactions_df.reset_index(drop=True)


def aggregate_feature_by_time_window(
    data: pd.DataFrame,
    windows_size_in_days: List[int],
    time_unit: TimeUnits,
    feature_name: str,
    agg_func_list: List[AggFunc],
    datetime_col: str,
    index_name: str,
) -> pd.DataFrame:
    """Aggregate a feature by the given time window.

    Args:
        data: pd.DataFrame
            Data frame to be aggregated
        windows_size_in_days: List[int]
            List of the window sizes to aggregate the feature.
        time_unit: TimeUnits
            Time unit for the window size.
        feature_name: str
            Name of the feature that will be transformed.
        agg_func_list: AggFunc
            List of the Aggregation functions to be applied.
        datetime_col: str
            Name of the timestamp column.
        index_name: str
            Name of the index.

    Returns:
        pd.Series:
            Feature aggregated.
    """
    data = data.sort_values(datetime_col)

    data = data.reset_index().set_index(keys=datetime_col)

    for window_size in windows_size_in_days:

        for agg_func in agg_func_list:
            aggregated_feature = (
                data[feature_name]
                .rolling(
                    window=pd.Timedelta(
                        value=window_size, unit=time_unit.value
                    )
                )
                .agg(agg_func.value)
            )

            data[
                "customer_"
                + agg_func.value
                + "_"
                + feature_name
                + "_"
                + str(window_size)
                + "_"
                + time_unit.value
            ] = list(aggregated_feature)

    data = data.reset_index().set_index(keys=index_name)

    return data
