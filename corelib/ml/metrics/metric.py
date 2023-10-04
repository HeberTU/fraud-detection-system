# -*- coding: utf-8 -*-
"""Metric Abstraction.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

from numpy.typing import NDArray


@dataclass
class Results:
    """Data structure for storing result the model.

    Parameters
    ----------
    predictions: NDArray
        The output of a NN model.
    scores: NDArray
        probability, for probabilistic models, decision function otherwise.
    """

    predictions: NDArray
    scores: NDArray


@dataclass
class TrueValues:
    """Data structure for storing true values.

    Parameters
    ----------
    tx_fraud: NDArray
        Whether the transaction was fraudulent
    tx_datetime: NDArray
        Timestamp for th transaction.
    customer_id: NDArray
        unique identifier for the customer.
    """

    tx_fraud: NDArray
    tx_datetime: NDArray
    customer_id: NDArray


@dataclass
class Metric(ABC):
    """ML model mMetric abstraction."""

    name: str = ""
    params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def measure(self, results: Results, true_values: TrueValues) -> float:
        """Measure the model performance.

        Args:
            results: Results
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.

        Returns:
            float:
                model performance score.
        """
        raise NotImplementedError
