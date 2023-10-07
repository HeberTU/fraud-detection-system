# -*- coding: utf-8 -*-
"""Hyperparameter configuration.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass


@dataclass
class HPOConfig:
    """HPO Configuration."""

    n_calls: int = 100
    n_random_starts: int = 10
    random_state: int = 19911127
