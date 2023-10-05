# -*- coding: utf-8 -*-
"""Fake Estimator.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

import numpy as np
import pandas as pd

from corelib.ml import metrics
from corelib.ml.estimators.estimator import Estimator
from corelib.ml.hyperparam_optim.search_dimension import (
    SKOptHyperparameterDimension,  # fmt: skip
)


class FakeEstimator(Estimator):
    """Fake estimator for testing."""

    def creat_model(self) -> Dict[str, Any]:
        """Create a model, ml pipeline logic.

        Returns:
            Dict[str, Any]:
                Tests results.
        """
        return {}

    def fit(
        self, data: pd.DataFrame, hyper_parameters: Dict[str, Any]
    ) -> None:
        """Fit ML algorithm.

        Args:
            data: pd.DataFrame
                Data that will be used to train the algorithm.
            hyper_parameters: Dict[str, Any]
                Hyper parameters.

        Returns:
            None
        """
        pass

    def predict(self, data: pd.DataFrame) -> metrics.Results:
        """Generate model predictions.

        Args:
            data: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            metrics.Results
        """
        return metrics.Results(
            predictions=np.array([1, 0]),
            scores=np.array([0.9, 0.1]),
        )

    def evaluate(self, data: pd.DataFrame, hashed_data: str) -> Dict[str, Any]:
        """Evaluate ml model.

        Args:
            data: pd.DataFrame
                Data that will be used to evaluate model, usually is the
                testing data.
            hashed_data: str

        Returns:
            Dict[str, float]
        """
        return {}

    def optimize_and_fit(
        self,
        data: pd.DataFrame,
        hpo_dimension: Dict[str, SKOptHyperparameterDimension],
    ):
        """Perform hyperparameter optimization and fit the final model.

        Args:
            data: pd.DataFrame
                Data that will be used to train the algorithm.
            hpo_dimension: Dict[str, skopt.space.Dimension]
                Hyperparameter dimensions.

        Returns:
            None
        """
        raise {}

    def hyperparameter_searcher(
        self,
        data: pd.DataFrame,
        hpo_dimension: Dict[str, SKOptHyperparameterDimension],
    ) -> Dict[str, Any]:
        """Perform hyperparameter search.

        Args:
            data: pd.DataFrame
                Training data.
            hpo_dimension: Dict[str, SKOptHyperparameterDimension]

        Returns:
            Dict[str, Any]:
                Best possible hyperparameter values.
        """
        return {}
