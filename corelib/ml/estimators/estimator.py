# -*- coding: utf-8 -*-
"""Estimator class.

This class will implement all the ML pipeline logic.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import datetime
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    Optional,
)

import pandas as pd

from corelib import (
    data_repositories,
    data_schemas,
)
from corelib.ml import metrics
from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.evaluators.evaluator import Evaluator
from corelib.ml.hyperparam_optim.search_dimension import (
    SKOptHyperparameterDimension,  # fmt: skip
)
from corelib.ml.transformers.transformer_chain import TransformerChain


class Estimator(ABC):
    """Estimator Interface."""

    def __init__(
        self,
        data_repository: Optional[data_repositories.DataRepository],
        evaluator: Optional[Evaluator],
        feature_schemas: data_schemas.BaseSchema,
        target_schema: Optional[data_schemas.BaseSchema],
        timestamp_schema: Optional[data_schemas.BaseSchema],
        customer_id_schema: Optional[data_schemas.BaseSchema],
        algorithm: Algorithm,
        transformer_chain: TransformerChain,
        do_hpo: bool,
    ):
        """Instantiate a Base Algorithm.

        Args:
            data_repository: Optional[data_repositories.DataRepository]
                Data repository to get the data. Optional for Inference
            evaluator: Optional[Evaluator]
                Ml model evaluator. . Optional for Inference.
            feature_schemas: data_schemas.BaseSchema
                Data schemas that defines feature space.
            target_schema: Optional[data_schemas.BaseSchema]
                Data schemas that defines target. Optional for Inference.
            timestamp_schema: Optional[data_schemas.BaseSchema]
                Time stamp schema for model evaluation.
            customer_id_schema: Optional[data_schemas.BaseSchema],
                customer_id schema for model evaluation.
            algorithm: BaseEstimator
                ML algorithm to tran and test.
            transformer_chain: TransformerChain
                Feature transformer.
            do_hpo: bool
                If True, the estimator will do hyperparameter search.
        """
        self.data_repository = data_repository
        self.evaluator = evaluator

        self.feature_schemas = feature_schemas
        self.target_schema = target_schema
        self.timestamp_schema = timestamp_schema
        self.customer_id_schema = customer_id_schema

        self.algorithm = algorithm
        self.transformer_chain = transformer_chain

        self._version = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        self.do_hpo = do_hpo

        self.hpo_dimension: Optional[
            Dict[str, SKOptHyperparameterDimension]
        ] = None

    @abstractmethod
    def creat_model(self) -> Dict[str, Any]:
        """Create a model, ml pipeline logic.

        Returns:
            Dict[str, Any]:
                Tests results.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(
        self, data: pd.DataFrame, hyper_parameters: Dict[str, Any]
    ) -> None:
        """Fit ML algorithm.

        Args:
            data: pd.DataFrame
                Data that will be used to train the algorithm.
            hyper_parameters: Dict[str, Any]
                Hyper parameters.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> metrics.Results:
        """Generate model predictions.

        Args:
            data: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            metrics.Results
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        hashed_data: str,
        plot_results: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate ml model.

        Args:
            data: pd.DataFrame
                Data that will be used to evaluate model, usually is the
                testing data.
            hashed_data: str
                Hash representation of the data.
            plot_results: bool
                If True, model results will be plotted.

        Returns:
            Dict[str, float]
        """
        raise NotImplementedError

    def set_model_artifacts(self, integration_test_set: pd.DataFrame) -> None:
        """Set model Artifacts.

        Args:
            integration_test_set: pd.DataFrame
                This data frame will be used to testing during deployment.

        Returns:
            None
        """
        integration_test_set["predicted"] = self.predict(
            data=integration_test_set
        ).scores

        artifact_repo = ArtifactRepo(
            feature_schemas=self.feature_schemas,
            transformer_chain=self.transformer_chain,
            algorithm=self.algorithm,
            integration_test_set=integration_test_set,
        )

        artifact_repo.dump_artifacts()

    @abstractmethod
    def optimize_and_fit(
        self,
        data: pd.DataFrame,
        hpo_dimension: Dict[str, SKOptHyperparameterDimension],
    ):
        """Perform hyperparameter optimization and fit the final model.

        Args:
            data: pd.DataFrame
                Data that will be used to train the algorithm.
            hpo_dimension: Dict[str, skopt.space.Dimension]
                Hyperparameter dimensions.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def hyperparameter_searcher(
        self,
        data: pd.DataFrame,
        hpo_dimension: Dict[str, SKOptHyperparameterDimension],
    ) -> Dict[str, Any]:
        """Perform hyperparameter search.

        Args:
            data: pd.DataFrame
                Training data.
            hpo_dimension: Dict[str, SKOptHyperparameterDimension]

        Returns:
            Dict[str, Any]:
                Best possible hyperparameter values.
        """
        raise NotImplementedError
