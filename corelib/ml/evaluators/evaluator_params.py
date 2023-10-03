# -*- coding: utf-8 -*-
"""Evaluator params.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass


@dataclass
class TimeEvaluatorParams:
    """Time evaluator parameters."""

    date_colum_name: str = "tx_datetime"
    delta_test_in_days: int = 7
    delta_delay_in_days: int = 7
