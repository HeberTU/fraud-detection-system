# -*- coding: utf-8 -*-
"""Decision Tree wraper.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(DecisionTreeClassifier):
    """Decision tree classifier wrapper."""

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
