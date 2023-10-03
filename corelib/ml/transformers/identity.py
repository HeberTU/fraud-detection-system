# -*- coding: utf-8 -*-
"""Identity transformer.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd

from corelib.ml.transformers.transformer import FeatureTransformer


class IdentityTransformer(FeatureTransformer):
    """Identity transformer."""

    def fit_transformation(self, features: pd.DataFrame) -> None:
        """Fit identity transformer.

        Args:
            features: pd.DataFrame
                Input features to fit the transformation.

        Returns:
            None
        """
        pass

    def apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Perform identity transformation.

        Args:
            features: pd.DataFrame
                Features to apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        return features

    def fit_apply_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply the transformation.

        Args:
            features: pd.DataFrame
                Features to fit and apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        return features
