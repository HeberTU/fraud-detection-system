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

import pandas as pd


class FeatureTransformer(ABC):
    """Machine learning algorithm."""

    @abstractmethod
    def fit_transformation(self, features: pd.DataFrame) -> None:
        """Wraps the fit method.

        Args:
            features: pd.DataFrame
                Input features to fit the transformation.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation.

        Args:
            features: pd.DataFrame
                Features to apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        raise NotImplementedError

    def fit_apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply the transformation.

        Args:
            features: pd.DataFrame
                Features to fit and apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        raise NotImplementedError
