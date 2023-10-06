# -*- coding: utf-8 -*-
"""Configuration file for managing pytest artifacts.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from pandera.typing import Series

from corelib.data_schemas.data_schema_factory import DataSchemaFactory
from corelib.entrypoints.api import app
from corelib.ml.algorithms.algorithm_factory import AlgorithmFactory
from corelib.ml.algorithms.algorithm_params import (
    LightGBMHPOParams,
    LightGBMParams,
)
from corelib.ml.transformers.transformers_factory import TransformerFactory
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
def client() -> httpx.AsyncClient:
    """Get API for testing."""
    return httpx.AsyncClient(app=app, base_url="http://test")


@pytest.fixture
def prediction_request() -> PredictionRequest:
    """Get prediction request instance."""
    return PredictionRequest(
        tx_amount=150.75,
        is_weekday=1,
        is_night=0,
        customer_id_mean_tx_amount_1_days=125.50,
        customer_id_count_tx_amount_1_days=3,
        customer_id_mean_tx_amount_7_days=110.25,
        customer_id_count_tx_amount_7_days=20,
        customer_id_mean_tx_amount_30_days=105.30,
        customer_id_count_tx_amount_30_days=85,
        terminal_id_mean_tx_fraud_1_days=0.05,
        terminal_id_mean_tx_fraud_7_days=0.03,
        terminal_id_mean_tx_fraud_30_days=0.02,
    )


@pytest.fixture
def ml_artifacts(request: FixtureRequest) -> Dict[str, Any]:
    """Create ml artifacts."""
    data_repository_type = request.param.get("data_repository_type")
    algorithm_type = request.param.get("algorithm_type")
    transformer_type = request.param.get("transformer_type")
    data_schemas = DataSchemaFactory().create(
        data_repository_type=data_repository_type
    )
    algorithm = AlgorithmFactory().create(algorithm_type=algorithm_type)
    feature_transformer = TransformerFactory().create(
        transformer_type=transformer_type
    )
    integration_test_set = pd.DataFrame()

    return {
        "feature_schemas": data_schemas,
        "feature_transformer": feature_transformer,
        "algorithm": algorithm,
        "integration_test_set": integration_test_set,
    }


@pytest.fixture
def mock_settings(tmp_path) -> None:
    """Mock the ASSETS_PATH with tmp_path for testing purposes."""
    with patch("corelib.config.settings.ASSETS_PATH", tmp_path):
        yield
