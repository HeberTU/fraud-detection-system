# -*- coding: utf-8 -*-
"""Search Dimension module.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Union,
)

import skopt


class Prior(str, enum.Enum):
    """Available Prior Distributions."""

    UNIFORM: Prior = "uniform"
    LOG_UNIFORM: Prior = "log-uniform"


class SKOptHyperparameterDimension(ABC):
    """SKO Hyperparameter dimension Abstraction."""

    @abstractmethod
    def skopt_form(self) -> skopt.space.Dimension:
        """Instantiate search space dimension."""
        raise NotImplementedError

    @abstractmethod
    def parse_value(
        self, value: Union[int, float, str]
    ) -> Union[int, float, str]:
        """Coerce value.

        Args:
            value: Union[int, float]
                Value to coerce.

        Returns:
            Coerced Value.
        """
        raise NotImplementedError


class IntegerDimension(SKOptHyperparameterDimension):
    """Integer Search Dimension."""

    def __init__(
        self, interval_start: int, interval_end: int, prior: Prior, name: str
    ):
        """Instantiate an integer search dimension.

        Args:
            interval_start: int
                Minimum search value.
            interval_end: int
                Maximum search value,
            prior: Prior
                Distribution to use when sampling random integers for this
                dimension.
            name: str
                hyperparameter name
        """
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.prior = prior
        self.name = name

    def skopt_form(self) -> skopt.space.Dimension:
        """Instantiate search space dimension."""
        return skopt.space.Integer(
            self.interval_start,
            self.interval_end,
            prior=self.prior.value,
            name=self.name,
        )

    def parse_value(
        self, value: Union[int, float, str]
    ) -> Union[int, float, str]:
        """Coerce value.

        Args:
            value: Union[int, float]
                Value to coerce.

        Returns:
            Coerced Value.
        """
        return int(value)

    @property
    def max_possible_value(self) -> int:
        """Get the maximum possible value."""
        return self.interval_end


class RealDimension(SKOptHyperparameterDimension):
    """Real (decimal) search dimension."""

    def __init__(
        self,
        interval_start: float,
        interval_end: float,
        prior: Prior,
        name: str,
    ):
        """Instantiate a real search dimension.

        Args:
            interval_start: float
                Minimum search value.
            interval_end: float
                Maximum search value,
            prior: Prior
                Distribution to use when sampling random integers for this
                dimension.
            name: str
                hyperparameter name
        """
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.prior = prior
        self.name = name

    def skopt_form(self):
        """Instantiate search space dimension."""
        return skopt.space.Real(
            self.interval_start,
            self.interval_end,
            prior=self.prior.value,
            name=self.name,
        )

    def parse_value(
        self, value: Union[int, float, str]
    ) -> Union[int, float, str]:
        """Coerce value.

        Args:
            value: Union[int, float]
                Value to coerce.

        Returns:
            Coerced Value.
        """
        return float(value)

    @property
    def max_possible_value(self) -> float:
        """Get the maximum possible value."""
        return self.interval_end


class CategoricalDimension(SKOptHyperparameterDimension):
    """Categorical search dimension."""

    def __init__(self, categories: List[str], name: str):
        """Instantiate a categorical search dimension.

        Args:
            categories: List[str]
                List of possible values
            name: str
                hyperparameter name.
        """
        self.categories = categories
        self.name = name

    def skopt_form(self) -> skopt.space.Dimension:
        """Instantiate search space dimension."""
        return skopt.space.Categorical(
            categories=self.categories, name=self.name
        )

    def parse_value(
        self, value: Union[int, float, str]
    ) -> Union[int, float, str]:
        """Coerce value.

        Args:
            value: Union[int, float, str]
                Value to coerce.

        Returns:
            Coerced Value.
        """
        return value

    @property
    def max_possible_value(self):
        """Get the maximum possible values."""
        return max(self.categories)


def get_dimensions(
    search_dimensions: Dict[str, SKOptHyperparameterDimension]
) -> List[skopt.space.Dimension]:
    """Transform the search dimension into a list.

    Args:
        search_dimensions:

    Returns:
        List[skopt.space.Dimension]
            Hyper-dimension search space in list format.
    """
    skopt_dimention_list = []
    for key in search_dimensions:
        skopt_dimention_list.append(search_dimensions[key].skopt_form())
    return skopt_dimention_list


def get_hyperparamrs_dict(
    search_dimensions: Dict[str, SKOptHyperparameterDimension],
    hyper_params_list: List[Any],
) -> Dict[str, Any]:
    """Get hyperparams names from a list of values.

    Args:
        search_dimensions: Dict[str, SKOptHyperparameterDimension]
            Dimension to optimize.
        hyper_params_list: List[Any]

    Returns:
        Dict[str, Any]
            Dictionary of hyperparameters.
    """
    hyperparameters_dict = {}
    i = 0
    for key in search_dimensions:
        hyperparameters_dict[key] = search_dimensions[key].parse_value(
            hyper_params_list[i]
        )
        i += 1
    return hyperparameters_dict
