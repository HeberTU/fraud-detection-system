# -*- coding: utf-8 -*-
"""Identity transformer.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Union

import pandas as pd

from corelib.ml.transformers.transformer import FeatureTransformer


class IdentityTransformer(FeatureTransformer):
    """Identity transformer."""

    def fit_transformation(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> None:
        """Fit identity transformer.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Input features to fit the transformation.

        Returns:
            None
        """
        pass

    def apply_transformation(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Perform identity transformation.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Features to apply transformation

        Returns:
            pd.DataFrame:
                Transformed features.
        """
        return features

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
        return features
