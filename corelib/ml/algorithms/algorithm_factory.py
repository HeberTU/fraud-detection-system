# -*- coding: utf-8 -*-
"""ML Algorithm's factory.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from corelib.ml.algorithms.algorithm_params import DecisionTreeParams


class AlgorithmType(str, enum.Enum):
    """Available ml algorithms."""

    DECISION_TREE: AlgorithmType = "DECISION_TREE"


class AlgorithmFactory:
    """Evaluator factory."""

    def __init__(self):
        """Initialize ml algorithm factory."""
        self._params = {AlgorithmType.DECISION_TREE: DecisionTreeParams}
        self._catalogue = {AlgorithmType.DECISION_TREE: DecisionTreeClassifier}

    def create(self, algorithm_type: AlgorithmType) -> BaseEstimator:
        """Instantiate an ML algorithm implementation.

        Args:
            algorithm_type: AlgorithmType
                algorithm type.

        Returns:
            BaseEstimator:
                ML algorithm instance.
        """
        params = self._params.get(algorithm_type, None)

        if params is None:
            raise NotImplementedError(
                f"{algorithm_type} parameters not implemented"
            )

        algorithm = self._catalogue.get(algorithm_type, None)

        if algorithm is None:
            raise NotImplementedError(f"{algorithm_type} not implemented")

        return algorithm(**params().__dict__)
