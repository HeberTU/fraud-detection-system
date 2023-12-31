# -*- coding: utf-8 -*-
"""Data repository parameters.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import (
    dataclass,
    field,
)

import pandas as pd


@dataclass
class SyntheticParams:
    """Synthetic data repository parameters."""

    n_customers: int = 5000
    n_terminals: int = 10000
    geo_uniform_lower_bound: int = 0
    geo_uniform_upper_bound: int = 100
    amount_uniform_lower_bound: int = 5
    amount_uniform_upper_bound: int = 100
    trans_uniform_lower_bound: int = 0
    trans_uniform_upper_bound: int = 4
    start_date: pd.Timedelta = field(
        default_factory=lambda: (
            pd.Timestamp("2023-09-30") - pd.Timedelta(value=183, unit="days")
        )
    )
    nb_days: int = 90
    radius: float = 5
    random_state: int = 0


@dataclass
class LocalParams:
    """Local data repository parameters."""

    file_name: str = "transactions.csv"
