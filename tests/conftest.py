# -*- coding: utf-8 -*-
"""Configuration file for managing pytest artifacts.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from pandera.typing import Series


@pytest.fixture()
def tx_datetime_series(request: FixtureRequest) -> Series[pd.Timestamp]:
    """datetime_series."""
    date_range_kwargs = request.param.get("date_range_kwargs")
    tx_datetime = pd.Series(pd.date_range(**date_range_kwargs))
    return pd.to_datetime(tx_datetime)
