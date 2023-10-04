# -*- coding: utf-8 -*-
"""Prediction service class.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib.ml.estimators.estimator import Estimator


class PredictionService:
    """Prediction Service Class."""

    def __init__(self, estimator: Estimator):
        """Instantiate the prediction service."""
        self.estimator = estimator

    def make_prediction(self, data) -> float:
        """Generate predictions using the estimator."""
        results = self.estimator.predict(data)
        return results.scores
