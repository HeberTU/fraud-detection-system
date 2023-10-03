# -*- coding: utf-8 -*-
"""Algorithm default parameters.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass


@dataclass
class DecisionTreeParams:
    """Time evaluator parameters."""

    criterion: str = "gini"
    max_depth: int = 2
    random_state: int = 0
