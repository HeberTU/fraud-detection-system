# -*- coding: utf-8 -*-
"""Decision Tree wraper.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier

from corelib.ml.algorithms.algorithm import Algorithm


class DecisionTree(DecisionTreeClassifier, Algorithm):
    """Decision tree classifier wrapper."""

    def fit_algorithm(
        self, features: pd.DataFrame, target: pd.DataFrame
    ) -> None:
        """Wraps the fit method.

        Args:
            features: pd.DataFrame
                Input features to fit the algorithm.
            target: pd.DataFrame
                Target Feature to fit the algorithm.

        Returns:
            None
        """
        self.fit(X=features, y=target)

    def get_predictions(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        return self.predict(X=features)

    def get_scores(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict log probs method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        log_probs = self.predict_log_proba(X=features)
        log_probs = log_probs[:, 1]
        log_probs = np.exp(log_probs)
        return log_probs

    def get_fit_param(self) -> Dict[str, Any]:
        """Get algorithm params.

        Returns:
            Dict[str, Any]:
                algorithm params.
        """
        algorithm_params = {
            "algorithm_name": "DecisionTree",
            **self.get_params(),
        }
        return algorithm_params
