# -*- coding: utf-8 -*-
"""Estimator class.

This class will implement all the ML pipeline logic.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

import pandas as pd

from corelib import (
    data_repositories,
    data_schemas,
    utils,
)
from corelib.ml import metrics
from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.evaluators.evaluator import Evaluator
from corelib.ml.transformers.transformer import FeatureTransformer


class Estimator:
    """ML Estimator."""

    def __init__(
        self,
        data_repository: data_repositories.DataRepository,
        evaluator: Evaluator,
        feature_schemas: data_schemas.BaseSchema,
        target_schema: data_schemas.BaseSchema,
        algorithm: Algorithm,
        feature_transformer: FeatureTransformer,
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
            feature_transformer: FeatureTransformer
                Feature transformer.
        """
        self.data_repository = data_repository
        self.evaluator = evaluator
        self.feature_schemas = feature_schemas
        self.target_schema = target_schema
        self.algorithm = algorithm
        self.feature_transformer = feature_transformer

        self._artifacts = {}

    def creat_model(self) -> Dict[str, Any]:
        """Create a model, ml pipeline logic.

        Returns:
            Dict[str, Any]:
                Tests results.
        """
        data = self.data_repository.load_data()
        processed_data = self.data_repository.preprocess(data=data)

        hashed_data = self.evaluator.hash_data(data=processed_data)

        train_data, test_data = self.evaluator.split(data=processed_data)

        self.fit(data=train_data)

        test_results = self.evaluate(
            data=test_data,
            hashed_data=hashed_data,
        )

        return test_results

    @utils.timer
    def fit(self, data: pd.DataFrame) -> None:
        """Fit ML algorithm.

        Args:
            data: pd.DataFrame
                Data that will be used to train the algorithm.

        Returns:
            None
        """
        features = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.feature_schemas
        )
        features = self.feature_transformer.fit_apply_transformation(
            features=features
        )
        target = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.target_schema
        )
        self.algorithm.fit_algorithm(features=features, target=target)

    def predict(self, data: pd.DataFrame) -> metrics.Results:
        """Generate model predictions.

        Args:
            data: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            metrics.Results
        """
        features = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.feature_schemas
        )
        features = self.feature_transformer.apply_transformation(
            features=features
        )
        return metrics.Results(
            predictions=self.algorithm.get_predictions(features=features),
            scores=self.algorithm.get_scores(features=features),
        )

    @utils.timer
    def evaluate(self, data: pd.DataFrame, hashed_data: str) -> Dict[str, Any]:
        """Evaluate ml model.

        Args:
            data: pd.DataFrame
                Data that will be used to evaluate model, usually is the
                testing data.
            hashed_data: str

        Returns:
            Dict[str, float]
        """
        results = self.predict(data=data)
        true_values = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.target_schema
        )
        results = self.evaluator.log_testing(
            estimator_params=self.algorithm.get_fit_param(),
            hashed_data=hashed_data,
            results=results,
            true_values=true_values,
        )
        return results
