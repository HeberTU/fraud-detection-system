# -*- coding: utf-8 -*-
"""Decision Tree wraper.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(DecisionTreeClassifier):
    """Decision tree classifier wrapper."""

    def get_predictions(self, data: pd.DataFrame) -> NDArray:
        """Wraps the predict method.

        Args:
            data: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        return self.predict(X=data)

    def get_scores(self, data: pd.DataFrame) -> NDArray:
        """Wraps the predict log probs method.

        Args:
            data: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        log_probs = self.predict_log_proba(X=data)
        return log_probs
