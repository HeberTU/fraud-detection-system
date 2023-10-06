# -*- coding: utf-8 -*-
"""Min max feature transformer.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from corelib.ml.transformers.transformer import FeatureTransformer


class MinMaxTransformer(MinMaxScaler, FeatureTransformer):
    """Machine learning algorithm."""

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
        self.fit(X=features)

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
        return self.transform(X=features)

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
        return self.fit_transform(X=features)
