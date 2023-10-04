# -*- coding: utf-8 -*-
"""ML algorithm interface.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
)

import pandas as pd
from numpy.typing import NDArray

from corelib.ml.hyperparam_optim import search_dimension


class Algorithm(ABC):
    """Machine learning algorithm."""

    def __init__(
        self,
        default_params: Dict[str, Any],
        hpo_params: Dict[str, search_dimension.SKOptHyperparameterDimension],
    ):
        """Instantiate a Light gbm wrapper.

        Args:
            default_params: Dict[str, Any]
                Default hyper-parameters.
            hpo_params: Dict[str, skopt.space.Dimension]
                Search dimensions for hpo.
        """
        self.params = default_params
        self.hpo_params = hpo_params

    @abstractmethod
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
                hyper parameters.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        raise NotImplementedError

    @abstractmethod
    def get_scores(self, features: pd.DataFrame) -> NDArray:
        """Wraps the predict probs method.

        Args:
            features: pd.DataFrame
                Data frame that will be used to generate predictions.

        Returns:
            NDArray:
                model predictions
        """
        raise NotImplementedError

    @abstractmethod
    def get_fit_param(self) -> Dict[str, Any]:
        """Get algorithm params.

        Returns:
            Dict[str, Any]:
                algorithm params.
        """
        raise NotImplementedError
