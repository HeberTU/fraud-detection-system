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
class Metric(ABC):
    """ML model mMetric abstraction."""

    name: str
    params: Optional[Dict[str, Any]] = None

    @abstractmethod
    def measure(self, predictions: NDArray, true_values: NDArray) -> float:
        """Measure the model performance.

        Args:
            predictions: NDArray
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.

        Returns:
            float:
                model performance score.
        """
        raise NotImplementedError
