# -*- coding: utf-8 -*-
"""Local Estimator Module.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib.ml.algorithms.algorithm_factory import AlgorithmType
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.estimators.estimator_factory import EstimatorFactory

local_estimator = EstimatorFactory.create_from_artifact_repo(
    artifact_repo=ArtifactRepo.load_from_assets(
        algorithm_type=AlgorithmType.LIGHT_GBM
    )
)
