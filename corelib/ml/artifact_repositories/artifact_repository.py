# -*- coding: utf-8 -*-
"""Artifact repository.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass

import pandas as pd

from corelib import (
    config,
    data_schemas,
    utils,
)
from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.transformers.transformer import FeatureTransformer


@dataclass
class ArtifactRepo:
    """ML Algorithm artifacts."""

    feature_schemas: data_schemas.BaseSchema
    feature_transformer: FeatureTransformer
    algorithm: Algorithm
    integration_test_set: pd.DataFrame()

    def dump_artifacts(self) -> None:
        """Dump artifact repo items."""
        file_path = (
            config.settings.ASSETS_PATH / self.algorithm.__class__.__name__
        )

        for key, value in self.__dict__.items():
            utils.dump_artifacts(
                obj=value, file_path=file_path, file_name=key + ".pickle"
            )
