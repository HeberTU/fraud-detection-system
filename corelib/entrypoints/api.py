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

from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.estimators.estimator import Estimator
from corelib.ml.estimators.estimator_factory import EstimatorFactory
from corelib.services.contracts import (
    PredictionRequest,
    PredictionResponse,
)
from corelib.services.prediction_service import PredictionService

estimator = EstimatorFactory.create_from_artifact_repo(
    artifact_repo=ArtifactRepo.load_from_assets(
        algorithm_type=AlgorithmType.LIGHT_GBM
    )
)

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    estimator: Estimator = Depends(lambda: estimator),
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
