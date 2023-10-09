# -*- coding: utf-8 -*-
"""Deployment tests.

Created on: 7/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories import ArtifactRepo


@pytest.mark.deployment
@pytest.mark.parametrize(
    "artifact_repo",
    [
        {
            "algorithm_type": AlgorithmType.LIGHT_GBM,
        }
    ],
    indirect=True,
)
def test_prediction_match_training(
    artifact_repo: ArtifactRepo, client_entrypoint: TestClient
):
    """Test deployment model."""
    integration_test_set = artifact_repo.integration_test_set
    integration_test_set = integration_test_set.reset_index()
    integration_test_set["tx_datetime"] = (
        pd.to_datetime(arg=integration_test_set["tx_datetime"]).astype(int)
        // 10**6
    )

    integration_test_dict = integration_test_set.to_dict(orient="records")

    for request_payload in integration_test_dict:

        response = client_entrypoint.post(
            f"/model/v0/prediction/{request_payload.get('transaction_id')}",
            json=request_payload,
        )

        assert response.status_code == 200
        response_json = response.json()

        assert "transaction_id" in response_json
        assert "transaction_to_block" in response_json
        assert request_payload.get("predicted") == int(
            response_json.get("transaction_to_block")
        )
