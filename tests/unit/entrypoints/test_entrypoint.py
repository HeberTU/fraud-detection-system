# -*- coding: utf-8 -*-
"""Unit testing for the entrypoint.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import httpx
import pytest

from corelib.entrypoints.api import estimator
from corelib.ml.estimators.estimator import Estimator
from corelib.services.contracts import PredictionRequest


@pytest.mark.unit
def test_initialization() -> None:
    """Test the estimator singleton is correctly instantiated."""
    assert isinstance(estimator, Estimator)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_endpoint(
    client: httpx.AsyncClient, prediction_request: PredictionRequest
) -> None:
    """Test predict endpoint."""
    response = await client.post("/predict", json=prediction_request)

    assert response.status_code == 200
    assert "prediction" in response.json()
