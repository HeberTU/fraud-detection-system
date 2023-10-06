# -*- coding: utf-8 -*-
"""Unit test suit for ArtifactRepo class.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)
from unittest.mock import patch

import pytest

from corelib.config.settings import Settings
from corelib.data_repositories.data_repository_factory import (
    DataRepositoryType,  # fmt: skip
)
from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories.artifact_repository import ArtifactRepo
from corelib.ml.transformers.transformers_factory import TransformerType


@pytest.mark.unit
@pytest.mark.parametrize(
    "ml_artifacts",
    [
        {
            "data_repository_type": DataRepositoryType.SYNTHETIC,
            "algorithm_type": AlgorithmType.LIGHT_GBM,
            "transformer_type": TransformerType.IDENTITY,
        }
    ],
    indirect=True,
)
def test_artifacts_attributes(
    ml_artifacts: Dict[str, Any], mock_settings: Settings
) -> None:
    """Test Artifact repo call dump_artifacts the right times."""
    artifact_repo = ArtifactRepo(**ml_artifacts)

    assert hasattr(artifact_repo, "feature_schemas")
    assert hasattr(artifact_repo, "feature_transformer")
    assert hasattr(artifact_repo, "algorithm")
    assert hasattr(artifact_repo, "integration_test_set")


@pytest.mark.unit
@pytest.mark.parametrize(
    "ml_artifacts",
    [
        {
            "data_repository_type": DataRepositoryType.SYNTHETIC,
            "algorithm_type": AlgorithmType.LIGHT_GBM,
        }
    ],
    indirect=True,
)
def test_dump_artifacts_call(
    ml_artifacts: Dict[str, Any], mock_settings: Settings
) -> None:
    """Test Artifact repo call dump_artifacts the right times."""
    artifact_repo = ArtifactRepo(**ml_artifacts)

    with patch(target="corelib.utils.dump_artifacts") as mock_dump:
        artifact_repo.dump_artifacts()
        assert mock_dump.call_count == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "ml_artifacts",
    [
        {
            "data_repository_type": DataRepositoryType.SYNTHETIC,
            "algorithm_type": AlgorithmType.LIGHT_GBM,
        }
    ],
    indirect=True,
)
def test_load_from_assets_call(
    ml_artifacts: Dict[str, Any], mock_settings: Settings
) -> None:
    """Test Artifact repo call dump_artifacts the right times."""
    algorithm_type = AlgorithmType.LIGHT_GBM

    with patch(
        target="corelib.utils.load_artifacts",
        side_effect=ml_artifacts.values(),
    ) as mock_load:
        _ = ArtifactRepo.load_from_assets(algorithm_type)
        assert mock_load.call_count == 4
