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

from corelib.config import settings
from corelib.entrypoints.assets import Assets
from corelib.ml.estimators.estimator import Estimator
from corelib.services.contracts import (
    PredictionRequest,
    PredictionResponse,
)
from corelib.services.prediction_service import PredictionService

estimator = Assets(settings.ENV)()

app = FastAPI()


@app.post(
    "/model/v0/prediction/{transaction_id}", response_model=PredictionResponse
)
def predict(
    transaction_id: str,
    request: PredictionRequest,
    estimator: Estimator = Depends(lambda: estimator),
) -> PredictionResponse:
    """Prediction endpoint.

    Args:
        transaction_id: str
            Unique ID for the transaction.
        request: ModifiedPredictionRequest
            Input features.
        estimator: Estimator
            Estimator class.

    Returns:
        ModifiedPredictionResponse:
            Decision to block the transaction or not.
    """
    service = PredictionService(estimator)
    prediction = service.make_prediction(request)

    return PredictionResponse(
        transaction_id=transaction_id, transaction_to_block=prediction
    )
