# -*- coding: utf-8 -*-
"""Light GBM algorithm.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
    Optional,
)

import lightgbm as lgb
import pandas as pd
from numpy.typing import NDArray

from corelib.ml.algorithms.algorithm import Algorithm


class LightGBM(Algorithm):
    """Light GBM classifier wrapper."""

    def __init__(
        self,
        objective: str,
        num_iterations: int,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        bagging_fraction: float,
        feature_fraction: float,
        min_gain_to_split: float,
        min_data_in_leaf: float,
        lambda_l1: float,
        lambda_l2: float,
        num_threads: int,
        verbose: int,
        threshold: float,
    ):
        """Instantiate a Light gbm wrapper.

        Args:
            objective: str
                Name of the loss function.
            num_iterations: int:
                Number of boosting iterations.
            max_depth: int
                Limit the max depth for tree model. This is used to deal with
                over-fitting when data is small.
            num_leaves: int
                Max number of leaves in one tree
            learning_rate: float
                Shrinkage rate
            feature_fraction: float
                LightGBM will randomly select a subset of features on each
                iteration (tree) if feature_fraction is smaller than 1.0. For
                example, if you set it to 0.8, LightGBM will select 80% of
                features before training each tree.
            bagging_fraction: float
                Like feature_fraction, but this will randomly select part of
                 data without resampling
            min_gain_to_split: float
                The minimal gain to perform split
            min_data_in_leaf: int
                Minimal number of data in one leaf. Can be used to deal with
                over-fitting
            lambda_l1: float
                L1 regularization
            lambda_l2: float
                L3 regularization
            num_threads: int
                Number of threads for LightGBM
            verbose: int
                Controls the level of LightGBM’s verbosity.
            threshold: float
                Probability threshold to evaluate true when predicting.
        """
        self.lgb_params = {
            "objective": objective,
            "num_iterations": int(num_iterations),
            "max_depth": int(max_depth),
            "num_leaves": int(num_leaves),
            "learning_rate": learning_rate,
            "bagging_fraction": bagging_fraction,
            "feature_fraction": feature_fraction,
            "min_gain_to_split": min_gain_to_split,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "num_threads": int(num_threads),
            "verbose": int(verbose),
        }
        self.gbm: Optional[lgb.Booster] = None
        self.threshold = threshold

    def fit_algorithm(
        self, features: pd.DataFrame, target: pd.DataFrame
    ) -> None:
        """Wraps the fit method.

        Args:
            features: pd.DataFrame
                Input features to fit the algorithm.
            target: pd.DataFrame
                Target Feature to fit the algorithm.

        Returns:
            None
        """
        gbm = lgb.train(
            params=self.lgb_params,
            train_set=lgb.Dataset(data=features, label=target),
        )
        self.gbm = gbm

    def get_predictions(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        return self.gbm.predict(data=features) > self.threshold

    def get_scores(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict log probs method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        return self.gbm.predict(data=features)

    def get_fit_param(self) -> Dict[str, Any]:
        """Get algorithm params.

        Returns:
            Dict[str, Any]:
                algorithm params.
        """
        algorithm_params = {
            "algorithm_name": "LightGBM",
            **self.lgb_params,
        }
        return algorithm_params
