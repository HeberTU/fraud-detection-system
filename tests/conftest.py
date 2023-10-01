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
def transactions_df(request: FixtureRequest) -> Series[pd.Timestamp]:
    """datetime_series."""
    date_range_kwargs = request.param.get("date_range_kwargs")
    tx_datetime = pd.Series(pd.date_range(**date_range_kwargs))
    tx_datetime = pd.to_datetime(tx_datetime)

    data = pd.DataFrame(
        data={"tx_datetime": tx_datetime, "tx_amount": range(len(tx_datetime))}
    )

    data["transaction_id"] = range(len(data))

    return data.set_index("transaction_id")
