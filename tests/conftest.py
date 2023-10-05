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


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Data that simulates the testing dataset."""
    data = {
        "customer_id": [1, 2, 3, 4, 5, 1, 2, 6, 7, 8],
        "scores": [0.9, 0.8, 0.7, 0.4, 0.1, 0.95, 0.85, 0.75, 0.5, 0.05],
        "tx_fraud": [1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "tx_datetime": [
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
        ],
    }
    data = pd.DataFrame(data)
    data["tx_datetime"] = pd.to_datetime(data["tx_datetime"])
    return pd.DataFrame(data)
