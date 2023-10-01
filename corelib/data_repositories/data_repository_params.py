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

    n_customers: int = 500
    n_terminals: int = 1000
    geo_uniform_lower_bound: int = 0
    geo_uniform_upper_bound: int = 100
    amount_uniform_lower_bound: int = 5
    amount_uniform_upper_bound: int = 100
    trans_uniform_lower_bound: int = 0
    trans_uniform_upper_bound: int = 4
    start_date: pd.Timedelta = field(
        default_factory=lambda: pd.Timestamp("2023-09-30")
    )
    nb_days: int = 6
    radius: float = 10
    random_state: int = 0
