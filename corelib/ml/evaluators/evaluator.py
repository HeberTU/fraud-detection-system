# -*- coding: utf-8 -*-
"""Evaluator Abstraction.

Created on: 2/10/23
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
    Tuple,
)

import pandas as pd
from numpy.typing import NDArray

from corelib import utils

logger = utils.get_logger()


class Evaluator(ABC):
    """Machine learning model evaluator."""

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data for training and testig.

        Args:
            data: pd.DataFrame
                Data to split in training and testing.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Training and testing data.
        """
        raise NotImplementedError

    @staticmethod
    def hash_data(data: pd.DataFrame) -> str:
        """Hash data for tracking purposes.

        Args:
            data: pd.DataFrame
                Data that will be used to train and test the model.
        Returns:
            str: hashed representation of the data.
        """
        return utils.make_obj_hash(obj=data, mode="full")

    def evaluate(
        self, predictions: NDArray, true_values: NDArray
    ) -> Dict[str, float]:
        """Evaluate model predictions.

        Args:
            predictions: NDArray
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.

        Returns:
            Dict[str, float]:
                Evaluation metrics.
        """
        result = predictions * true_values
        return {"metric": result}

    def log_testing(
        self,
        estimator_params: Dict[str, Any],
        hashed_data: str,
        predictions: NDArray,
        true_values: NDArray,
    ) -> Dict[str, Any]:
        """Log model evaluation.

        Args:
            estimator_params: Dict[str, Any]
                Estimator parameters.
            hashed_data: str
                Hashed representation of the data.
            predictions: NDArray
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.

        Returns:
            Dict[str, Any]:
                Model results.

        """
        metrics = self.evaluate(
            predictions=predictions, true_values=true_values
        )
        results = {
            "metrics": metrics,
            "estimator_params": estimator_params,
            "hashed_data": hashed_data,
        }
        logger.info(results)
        return results
