# -*- coding: utf-8 -*-
"""Estimator class.

This class will implement all the ML pipeline logic.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import pandas as pd
import skopt

from corelib import (
    data_repositories,
    data_schemas,
    utils,
)
from corelib.ml import metrics
from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.artifact_repositories import ArtifactRepo
from corelib.ml.evaluators.evaluator import Evaluator
from corelib.ml.hyperparam_optim.hpo_config import HPOConfig
from corelib.ml.hyperparam_optim.search_dimension import (
    SKOptHyperparameterDimension,
    get_dimensions,
    get_hyperparamrs_dict,
)
from corelib.ml.transformers.transformer import FeatureTransformer


class Estimator:
    """ML Estimator."""

    def __init__(
        self,
        data_repository: Optional[data_repositories.DataRepository],
        evaluator: Optional[Evaluator],
        feature_schemas: data_schemas.BaseSchema,
        target_schema: Optional[data_schemas.BaseSchema],
        timestamp_schema: Optional[data_schemas.BaseSchema],
        customer_id_schema: Optional[data_schemas.BaseSchema],
        algorithm: Algorithm,
        feature_transformer: FeatureTransformer,
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
            feature_transformer: FeatureTransformer
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
        self.feature_transformer = feature_transformer

        self._version = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        self.do_hpo = do_hpo

        self.hpo_dimension: Optional[
            Dict[str, SKOptHyperparameterDimension]
        ] = None

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

        self.optimize_and_fit(
            data=train_data, hpo_dimension=self.algorithm.hpo_params
        )

        test_results = self.evaluate(
            data=test_data,
            hashed_data=hashed_data,
        )

        self.set_model_artifacts(integration_test_set=test_data.sample(n=1000))

        return test_results

    @utils.timer
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
        features = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.feature_schemas
        )
        features = self.feature_transformer.fit_apply_transformation(
            features=features
        )
        target = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.target_schema
        )
        self.algorithm.fit_algorithm(
            features=features, target=target, hyper_parameters=hyper_parameters
        )

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
        tx_datetime = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.timestamp_schema
        )
        customer_id = data_schemas.validate_and_coerce_schema(
            data=data, schema_class=self.customer_id_schema
        )
        results = self.evaluator.log_testing(
            estimator_params=self.algorithm.get_fit_param(),
            hashed_data=hashed_data,
            results=results,
            true_values=metrics.TrueValues(
                tx_fraud=true_values,
                tx_datetime=tx_datetime,
                customer_id=customer_id,
            ),
        )
        return results

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
            feature_transformer=self.feature_transformer,
            algorithm=self.algorithm,
            integration_test_set=integration_test_set,
        )

        artifact_repo.dump_artifacts()

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
        if self.do_hpo:
            self.algorithm.params = self.hyperparameter_searcher(
                data=data, hpo_dimension=hpo_dimension
            )

        self.fit(data=data, hyper_parameters=self.algorithm.params)

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
        self.hpo_dimension = hpo_dimension

        space = get_dimensions(search_dimensions=hpo_dimension)

        def val(hyper_parameters_list: List[Any]) -> float:
            """Wrapper function."""
            return self.validation_pipeline(
                data, hyper_parameters_list=hyper_parameters_list
            )

        res_gp = skopt.gp_minimize(
            func=val,
            dimensions=space,
            n_calls=HPOConfig.n_calls,
            n_random_starts=HPOConfig.n_random_starts,
            random_state=HPOConfig.random_state,
        )

        best_params = get_hyperparamrs_dict(
            search_dimensions=hpo_dimension, hyper_params_list=res_gp.x
        )

        return best_params

    def validation_pipeline(
        self, data: pd.DataFrame, hyper_parameters_list: List[Any]
    ) -> float:
        """Validate set of hyperparameters.

        Args:
            data: pd.DataFrame
                Training data.
            hyper_parameters_list: List[Any]
                List of hyperparameters to validate.

        Returns:
            float:
                score value.
        """
        hyper_parameters = get_hyperparamrs_dict(
            search_dimensions=self.hpo_dimension,
            hyper_params_list=hyper_parameters_list,
        )
        train_data, validation_data = self.evaluator.split(data=data)

        self.fit(data=train_data, hyper_parameters=hyper_parameters)

        validation_results = self.evaluate(
            data=validation_data,
            hashed_data="HPO",
        )

        return validation_results.get("scores").get("average_precision_score")
