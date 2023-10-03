# -*- coding: utf-8 -*-
"""Identity transformer.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from numpy.typing import NDArray
from sklearn.base import TransformerMixin


class IdentityTransformer(TransformerMixin):
    """Identity transformer."""

    def fit(self, X: NDArray) -> None:
        """Fit identity transformer.

        Args:
            X:  NDArray
                Data to fit the transformation on.

        Returns:
            None
        """
        pass

    def transform(self, X: NDArray) -> NDArray:
        """Perform identity transformation.

        Args:
            X: NDArray
                Data to transform.

        Returns:
            NDArray:
                Transformed data.
        """
        return X
