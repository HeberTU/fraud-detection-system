# -*- coding: utf-8 -*-
"""Artifact repository.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from corelib import (
    config,
    data_schemas,
    utils,
)
from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.transformers.transformer_chain import TransformerChain


@dataclass
class ArtifactRepo:
    """ML Algorithm artifacts."""

    feature_schemas: data_schemas.BaseSchema
    transformer_chain: TransformerChain
    algorithm: Algorithm
    integration_test_set: pd.DataFrame

    def dump_artifacts(self) -> None:
        """Dump artifact repo items."""
        file_path = (
            config.settings.ASSETS_PATH / self.algorithm.__class__.__name__
        )

        for key, value in self.__dict__.items():
            utils.dump_artifacts(
                obj=value, file_path=file_path, file_name=key + ".pickle"
            )

    @classmethod
    def load_from_assets(cls, algorithm_type: AlgorithmType) -> ArtifactRepo:
        """Loads an instance of ArtifactRepo from stored assets.

        Args:
            algorithm_type: AlgorithmType
                Algorithm artifacts that will be loaded.

        Returns:
            ArtifactRepo:
                Artifact repo.
        """
        files_path = config.settings.ASSETS_PATH / algorithm_type.value
        return cls(
            feature_schemas=utils.load_artifacts(
                file_path=files_path / "feature_schemas.pickle"
            ),
            transformer_chain=utils.load_artifacts(
                file_path=files_path / "transformer_chain.pickle"
            ),
            algorithm=utils.load_artifacts(
                file_path=files_path / "algorithm.pickle"
            ),
            integration_test_set=utils.load_artifacts(
                file_path=files_path / "integration_test_set.pickle"
            ),
        )
