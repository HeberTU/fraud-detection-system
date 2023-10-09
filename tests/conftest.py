# -*- coding: utf-8 -*-
"""Configuration file for managing pytest artifacts.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from datetime import datetime
from typing import (
    Any,
    Dict,
)
from unittest.mock import patch

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from fastapi.testclient import TestClient
from pandera.typing import Series

from corelib.data_schemas.data_schema_factory import DataSchemaFactory
from corelib.entrypoints.api import get_app
from corelib.ml.algorithms.algorithm_factory import AlgorithmFactory
from corelib.ml.algorithms.algorithm_params import (
    LightGBMHPOParams,
    LightGBMParams,
)
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.transformers.transformer_chain import TransformerChainFactory
from corelib.services.contracts import PredictionRequest


@pytest.fixture()
def transactions_df(request: FixtureRequest) -> Series[pd.Timestamp]:
    """datetime_series."""
    date_range_kwargs = request.param.get("date_range_kwargs")
    tx_datetime = pd.Series(pd.date_range(**date_range_kwargs))
    tx_datetime = pd.to_datetime(tx_datetime)

    data = pd.DataFrame(
        data={"tx_datetime": tx_datetime, "tx_amount": range(len(tx_datetime))}
    )

    data["transaction_id"] = range(len(data))

    return data.set_index("transaction_id")


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Data that simulates the testing dataset."""
    data = {
        "customer_id": [1, 2, 3, 4, 5, 1, 2, 6, 7, 8],
        "scores": [0.9, 0.8, 0.7, 0.4, 0.1, 0.95, 0.85, 0.75, 0.5, 0.05],
        "tx_fraud": [1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "tx_datetime": [
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
        ],
    }
    data = pd.DataFrame(data)
    data["tx_datetime"] = pd.to_datetime(data["tx_datetime"])
    return pd.DataFrame(data)


@pytest.fixture
def algorithm_artifacts() -> Dict[str, Any]:
    """Algorithm artifacts."""
    features = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]}
    )
    target = pd.DataFrame({"target": [1, 0, 1, 0, 1]})

    return {
        "features": features,
        "target": target,
        "default_params": LightGBMParams(),
        "hpo_params": LightGBMHPOParams(),
    }


@pytest.fixture
def prediction_request() -> PredictionRequest:
    """Get prediction request instance."""
    return PredictionRequest(
        tx_datetime=int(
            datetime.now().timestamp() * 1000
        ),  # current time as Unix timestamp in milliseconds
        tx_amount=150.75,
        customer_id_mean_tx_amount_1_days=125.50,
        customer_id_mean_tx_amount_7_days=110.25,
        customer_id_mean_tx_amount_30_days=105.30,
        customer_id_count_tx_amount_1_minutes=3,
        customer_id_count_tx_amount_5_minutes=20,
        customer_id_count_tx_amount_10_minutes=85,
        sector_id_mean_tx_fraud_1_days=0.05,
        sector_id_mean_tx_fraud_7_days=0.03,
        sector_id_mean_tx_fraud_30_days=0.02,
        customer_id_mean_tx_fraud_1_days=0.04,
        customer_id_mean_tx_fraud_7_days=0.02,
        customer_id_mean_tx_fraud_30_days=0.01,
    )


@pytest.fixture
def ml_artifacts(request: FixtureRequest) -> Dict[str, Any]:
    """Create ml artifacts."""
    data_repository_type = request.param.get("data_repository_type")
    algorithm_type = request.param.get("algorithm_type")
    data_schemas = DataSchemaFactory().create(
        data_repository_type=data_repository_type
    )
    algorithm = AlgorithmFactory().create(algorithm_type=algorithm_type)
    transformer_chain = TransformerChainFactory().create(
        data_repository_type=data_repository_type
    )
    integration_test_set = pd.DataFrame()

    return {
        "feature_schemas": data_schemas,
        "transformer_chain": transformer_chain,
        "algorithm": algorithm,
        "integration_test_set": integration_test_set,
    }


@pytest.fixture
def mock_settings(tmp_path) -> None:
    """Mock the ASSETS_PATH with tmp_path for testing purposes."""
    with patch("corelib.config.settings.ASSETS_PATH", tmp_path):
        yield


@pytest.fixture
def artifact_repo(request: FixtureRequest) -> ArtifactRepo:
    """Get artifact repo for deployment."""
    algorithm_type = request.param.get("algorithm_type")
    return ArtifactRepo.load_from_assets(algorithm_type=algorithm_type)


@pytest.fixture
def client_test() -> TestClient:
    """Instantiate a test client."""
    return TestClient(get_app())
