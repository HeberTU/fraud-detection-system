# -*- coding: utf-8 -*-
"""Algorithm default parameters.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass


@dataclass
class DecisionTreeParams:
    """Decision tree parameters."""

    criterion: str = "gini"
    max_depth: int = 2
    random_state: int = 0


@dataclass
class LightGBMParams:
    """Light GBM parameters."""

    objective: str = "xentropy"
    num_iterations: int = 100
    max_depth: int = 3
    num_leaves: int = 30
    learning_rate: float = 0.05
    bagging_fraction: float = 1
    feature_fraction: float = 1
    min_gain_to_split: float = 1
    min_data_in_leaf: float = 1
    lambda_l1: float = 0
    lambda_l2: float = 1
    num_threads: int = -1
    verbose: int = -1
    threshold: float = 0.5
