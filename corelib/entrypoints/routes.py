# -*- coding: utf-8 -*-
"""API routes module.

Created on: 9/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from fastapi import (
    APIRouter,
    Depends,
)

from corelib.config import settings
from corelib.entrypoints.assets import Assets
from corelib.ml.estimators.estimator import Estimator
from corelib.services.contracts import (
    PredictionRequest,
    PredictionResponse,
)
from corelib.services.prediction_service import PredictionService


def get_estimator() -> Estimator:
    """Get Estimator."""
    return Assets(settings.ENV)()


def get_router() -> APIRouter:
    """Get API Router."""
    router = APIRouter()

    @router.post(
        path="/model/v0/prediction/{transaction_id}",
        response_model=PredictionResponse,
    )
    def predict(
        transaction_id: str,
        request: PredictionRequest,
        estimator: Estimator = Depends(get_estimator),
    ) -> PredictionResponse:
        """Prediction endpoint.

        Args:
            transaction_id: str - Unique ID for the transaction.
            request: ModifiedPredictionRequest - Input features.
            estimator: Estimator - Estimator class.

        Returns:
            ModifiedPredictionResponse: Decision to block the transaction or
            not.
        """
        service = PredictionService(estimator)
        prediction = service.make_prediction(
            prediction_request=request, transaction_id=int(transaction_id)
        )

        return PredictionResponse(
            transaction_id=transaction_id, transaction_to_block=prediction
        )

    return router
