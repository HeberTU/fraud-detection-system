# -*- coding: utf-8 -*-
"""Local Estimator Module.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from corelib.config import Environment
from corelib.data_repositories.data_repository_factory import (
    DataRepositoryType,  # fmt: skip
)
from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.estimators.estimator import Estimator
from corelib.ml.estimators.estimator_factory import (
    EstimatorFactory,
    EstimatorType,
)
from corelib.ml.evaluators.evaluator_factory import EvaluatorType
from corelib.ml.transformers.transformers_factory import TransformerType


class Assets:
    """Returns the assets configuration depending on ENV variable."""

    def __init__(self, env: Environment):
        """Init Assets Factory."""
        self.env = env
        self._assets = {
            Environment.PROD.value: self.get_prod_assets,
            Environment.TEST.value: self.get_test_assets,
        }

    def __call__(self) -> Estimator:
        """Load assets depending on environment."""
        return self._assets.get(self.env)()

    @staticmethod
    def get_test_assets() -> Estimator:
        """Get tests assets."""
        return EstimatorFactory().create(
            estimator_type=EstimatorType.FAKE_ESTIMATOR,
            data_repository_type=DataRepositoryType.SYNTHETIC,
            evaluator_type=EvaluatorType.TIME_EVALUATOR,
            algorithm_type=AlgorithmType.LIGHT_GBM,
            transformer_type=TransformerType.MIN_MAX_SCALER,
            do_hpo=False,
        )

    @staticmethod
    def get_prod_assets() -> Estimator:
        """Get prod assets."""
        return EstimatorFactory().create_from_artifact_repo(
            estimator_type=EstimatorType.ML_ESTIMATOR,
            artifact_repo=ArtifactRepo.load_from_assets(
                algorithm_type=AlgorithmType.LIGHT_GBM
            ),
        )
