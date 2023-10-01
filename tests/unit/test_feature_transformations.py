# -*- coding: utf-8 -*-
"""Unit test suit for data transformations.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest
from pandera.typing import Series

from corelib.domain import feature_transformations


@pytest.mark.unit
@pytest.mark.parametrize(
    "tx_datetime_series",
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
def test_is_weeekday(tx_datetime_series: Series[pd.Timestamp]) -> None:
    """Test is_weeekday function."""
    result = feature_transformations.is_weekday(tx_datetime=tx_datetime_series)
    assert (result == pd.Series([1, 0, 0, 0, 0, 0, 1])).sum() == 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "tx_datetime_series",
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
def test_is_night(tx_datetime_series: Series[pd.Timestamp]) -> None:
    """Test is_night function."""
    results = feature_transformations.is_night(tx_datetime=tx_datetime_series)
    assert results.sum() == 7
