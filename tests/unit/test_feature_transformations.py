# -*- coding: utf-8 -*-
"""Unit test suit for data transformations.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest

from corelib.domain import feature_transformations


@pytest.mark.unit
@pytest.mark.parametrize(
    "transactions_df",
    [
        {
            "date_range_kwargs": {
                "start": pd.to_datetime("2023-10-01"),
                "end": pd.to_datetime("2023-10-07"),
                "periods": 7,
            }
        }
    ],
    indirect=True,
)
def test_is_weeekday(transactions_df: pd.DataFrame) -> None:
    """Test is_weeekday function."""
    result = feature_transformations.is_weekday(
        tx_datetime=transactions_df["tx_datetime"]
    )
    assert (result == pd.Series([1, 0, 0, 0, 0, 0, 1])).sum() == 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "transactions_df",
    [
        {
            "date_range_kwargs": {
                "start": pd.to_datetime("2023-10-01"),
                "periods": 10,
                "freq": "H",
            }
        }
    ],
    indirect=True,
)
def test_is_night(transactions_df: pd.DataFrame) -> None:
    """Test is_night function."""
    results = feature_transformations.is_night(
        tx_datetime=transactions_df["tx_datetime"]
    )
    assert results.sum() == 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "transactions_df",
    [
        {
            "date_range_kwargs": {
                "start": pd.to_datetime("2023-10-01"),
                "end": pd.to_datetime("2023-10-07"),
                "periods": 7,
            }
        }
    ],
    indirect=True,
)
def test_aggregate_feature_by_time_window(
    transactions_df: pd.DataFrame,
) -> None:
    """Test aggregate_feature_by_time_window."""
    transactions_df = feature_transformations.aggregate_feature_by_time_window(
        data=transactions_df,
        windows_size_in_days=[3],
        time_unit=feature_transformations.TimeUnits.DAYS,
        feature_name="tx_amount",
        agg_func_list=[
            feature_transformations.AggFunc.COUNT,
            feature_transformations.AggFunc.MEAN,
        ],
        datetime_col="tx_datetime",
        index_name="transaction_id",
    )

    assert transactions_df["customer_count_tx_amount_3_days"].sum() == 18
    assert transactions_df[
        "customer_count_tx_amount_3_days"
    ].mean() == pytest.approx(2.571, 0.001)
    assert transactions_df["customer_mean_tx_amount_3_days"].sum() == 15.5
    assert transactions_df[
        "customer_mean_tx_amount_3_days"
    ].mean() == pytest.approx(2.214, 0.001)
