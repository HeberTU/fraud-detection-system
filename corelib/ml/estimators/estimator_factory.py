# -*- coding: utf-8 -*-
"""ML Estimator Factory.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib import data_repositories as dr
from corelib import data_schemas as ds
from corelib.ml.algorithms.algorithm_factory import (
    AlgorithmFactory,
    AlgorithmType,
)
from corelib.ml.estimators.estimator import Estimator
from corelib.ml.evaluators.evaluator_factory import (
    EvaluatorFactory,
    EvaluatorType,
)
from corelib.ml.transformers.transformers_factory import (
    TransformerFactory,
    TransformerType,
)


class EstimatorFactory:
    """Estimator factory."""

    @staticmethod
    def create(
        data_repository_type: dr.DataRepositoryType,
        evaluator_type: EvaluatorType,
        algorithm_type: AlgorithmType,
        transformer_type: TransformerType,
    ) -> Estimator:
        """Instantiate the estimator.

        Args:
            data_repository_type: data_repositories.DataRepositoryType
                Type of data repository to get the data.
            evaluator_type: EvaluatorType
                Type of Ml model evaluator.
            algorithm_type: AlgorithmType
                Type of ML algorithm to tran and test.
            transformer_type: TransformerType
                Type of input transformer to be used.

        Returns:
            Evaluator
        """
        data_repository = dr.DataRepositoryFactory().create(
            data_repository_type=data_repository_type
        )
        evaluator = EvaluatorFactory().create(evaluator_type=evaluator_type)
        data_schemas = ds.DataSchemaFactory().create(
            data_repository_type=data_repository_type
        )
        algorithm = AlgorithmFactory().create(algorithm_type=algorithm_type)
        feature_transformer = TransformerFactory().create(
            transformer_type=transformer_type
        )

        return Estimator(
            data_repository=data_repository,
            evaluator=evaluator,
            feature_schemas=data_schemas.get("feature_space"),
            target_schema=data_schemas.get("target"),
            timestamp_schema=data_schemas.get("timestamp"),
            customer_id_schema=data_schemas.get("customer_id"),
            algorithm=algorithm,
            feature_transformer=feature_transformer,
        )
