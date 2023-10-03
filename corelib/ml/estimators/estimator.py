# -*- coding: utf-8 -*-
"""Estimator class.

This class will implement all the ML pipeline logic.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from sklearn.base import BaseEstimator

from corelib import (
    data_repositories,
    data_schemas,
)
from corelib.ml.evaluators.evaluator import Evaluator


class Estimator:
    """ML Estimator."""

    def __init__(
        self,
        data_repository: data_repositories.DataRepository,
        evaluator: Evaluator,
        feature_schemas: data_schemas.BaseSchema,
        target_schema: data_schemas.BaseSchema,
        algorithm: BaseEstimator,
    ):
        """Instantiate a Base Algorithm.

        Args:
            data_repository: data_repositories.DataRepository
                Data repository to get the data.
            evaluator: ml.Evaluator
                Ml model evaluator.
            feature_schemas: data_schemas.BaseSchema
                Data schemas that defines feature space.
            target_schema: data_schemas.BaseSchema
                Data schemas that defines target.
            algorithm: BaseEstimator
                ML algorithm to tran and test.
        """
        self.data_repository = data_repository
        self.evaluator = evaluator
        self.feature_schemas = feature_schemas
        self.target_schema = target_schema
        self.algorithm = algorithm

    def creat_model(self) -> None:
        """Create a model, ml pipeline logic.

        Returns:
            None
        """
        data = self.data_repository.load_data()
        processed_data = self.data_repository.preprocess(data=data)

        self.evaluator.split(data=processed_data)
