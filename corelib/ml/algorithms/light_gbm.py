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
import skopt
from numpy.typing import NDArray

from corelib.ml.algorithms.algorithm import Algorithm


class LightGBM(Algorithm):
    """Light GBM classifier wrapper."""

    def __init__(
        self,
        default_params: Dict[str, Any],
        hpo_params: Dict[str, skopt.space.Dimension],
        num_threads: int = -1,
        verbose: int = -1,
        threshold: float = 0.7,
    ):
        """Instantiate a Light gbm wrapper.

        Args:
            default_params: Dict[str, Any]
                Default hyper-parameters.
            hpo_params: Dict[str, skopt.space.Dimension]
                Search dimensions for hpo.
            num_threads: int
                Number of threads for LightGBM
            verbose: int
                Controls the level of LightGBM’s verbosity.
            threshold: float
                Probability threshold to evaluate true when predicting.
        """
        self.params = default_params.__dict__
        self.hpo_params = hpo_params.__dict__

        self.num_threads = num_threads
        self.verbose = verbose
        self.threshold = threshold

        self.gbm: Optional[lgb.Booster] = None

    def fit_algorithm(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        hyper_parameters: Dict[str, Any],
    ) -> None:
        """Wraps the fit method.

        Args:
            features: pd.DataFrame
                Input features to fit the algorithm.
            target: pd.DataFrame
                Target Feature to fit the algorithm.
            hyper_parameters: Dict[str, Any]
                Hyperparameters.

        Returns:
            None
        """
        hyper_parameters["num_threads"] = self.num_threads
        hyper_parameters["verbose"] = self.verbose
        gbm = lgb.train(
            params=hyper_parameters,
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
            **self.params,
        }
        return algorithm_params
