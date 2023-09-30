# -*- coding: utf-8 -*-
"""Data repository parameters.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass


@dataclass
class SyntheticParams:
    """Synthetic data repository parameters."""

    geo_uniform_lower_bound: int
    geo_uniform_upper_bound: int
    amount_uniform_lower_bound: int
    amount_uniform_upper_bound: int
    trans_uniform_lower_bound: int
    trans_uniform_upper_bound: int
