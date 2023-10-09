# -*- coding: utf-8 -*-
"""Deployment tests.

Created on: 7/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from pytest_benchmark.fixture import BenchmarkFixture

from corelib import utils
from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.services.contracts import PredictionRequest

logger = utils.get_logger()


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
    artifact_repo: ArtifactRepo, client_test: TestClient
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

        response = client_test.post(
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


@pytest.mark.benchmark(group="standard_invocation", max_time=5)
def test_latency_sla(
    benchmark: BenchmarkFixture,
    server: None,
    prediction_request: PredictionRequest,
) -> None:
    """Test latency for SLA."""
    request_payload = prediction_request.__dict__

    def post_request() -> None:
        """Invoke API."""
        requests.post(
            url="http://127.0.0.1:8000/model/v0/prediction/1",
            json=request_payload,
        )

    benchmark(post_request)

    average_latency_threshold = 0.07
    logger.info(f"Mean Latency: {benchmark.stats.stats.mean}")
    logger.info(f"Max Latency: {benchmark.stats.stats.max}")
    if benchmark.stats.stats.mean > average_latency_threshold:
        msg = (
            "Failed normal benchmark - Average latency higher "
            f"than {average_latency_threshold * 1000}ms"
        )
        pytest.fail(msg)
    max_latency_threshold = 0.5
    if benchmark.stats.stats.max > max_latency_threshold:
        msg = (
            f"Failed normal benchmark - Max latency higher "
            f"than {max_latency_threshold*1000}ms"
        )
        pytest.fail(msg)
