# -*- coding: utf-8 -*-
"""Standard feature transformer.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from corelib.ml.transformers.transformer import FeatureTransformer


class StandardTransformer(StandardScaler, FeatureTransformer):
    """Machine learning algorithm."""

    def fit_transformation(self, features: pd.DataFrame) -> None:
        """Wraps the fit method.

        Args:
            features: pd.DataFrame
                Input features to fit the transformation.

        Returns:
            None
        """
        self.fit(X=features)

    def apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation.

        Args:
            features: pd.DataFrame
                Features to apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        return self.transform(X=features)

    def fit_apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply the transformation.

        Args:
            features: pd.DataFrame
                Features to fit and apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        return self.fit_transform(X=features)
