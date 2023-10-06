# -*- coding: utf-8 -*-
"""Feature transformer interface.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)
from typing import Union

import pandas as pd


class FeatureTransformer(ABC):
    """Machine learning algorithm."""

    @abstractmethod
    def fit_transformation(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> None:
        """Wraps the fit method.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Input features to fit the transformation.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transformation(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Apply the transformation.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Features to apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        raise NotImplementedError

    def fit_apply_transformation(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Fit and apply the transformation.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Features to fit and apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        raise NotImplementedError
