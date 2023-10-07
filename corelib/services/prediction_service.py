# -*- coding: utf-8 -*-
"""Prediction service class.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
from fastapi.encoders import jsonable_encoder

from corelib.ml.estimators.estimator import Estimator
from corelib.services.contracts import PredictionRequest


class PredictionService:
    """Prediction Service Class."""

    def __init__(self, estimator: Estimator):
        """Instantiate the prediction service."""
        self.estimator = estimator

    def make_prediction(self, prediction_request: PredictionRequest) -> float:
        """Generate predictions using the estimator.

        Args:
            prediction_request: PredictionRequest
                Input contract.

        Returns:
            float
                prediction.
        """
        data = pd.DataFrame(jsonable_encoder(prediction_request), index=[0])
        data.tx_datetime = pd.Timestamp(data.tx_datetime, unit="ms")
        results = self.estimator.predict(data=data)
        return results.scores
