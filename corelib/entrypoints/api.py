# -*- coding: utf-8 -*-
"""API entrypoint.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from fastapi import (
    Depends,
    FastAPI,
)

from corelib.entrypoints.local_estimators import local_estimator
from corelib.ml.estimators.estimator import Estimator
from corelib.services.contracts import (
    PredictionRequest,
    PredictionResponse,
)
from corelib.services.prediction_service import PredictionService

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    estimator: Estimator = Depends(lambda: local_estimator),
) -> PredictionResponse:
    """Prediction endpoint.

    Args:
        request: PredictionRequest
            Input features.
        estimator: Estimator
            Estimator class.

    Returns:
        PredictionResponse:
            Fraud prediction.
    """
    service = PredictionService(estimator)
    prediction = service.make_prediction(request)

    return PredictionResponse(prediction=prediction)
