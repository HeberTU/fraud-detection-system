# -*- coding: utf-8 -*-
"""Feature Transformers factory.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

from corelib.ml.transformers.identity import IdentityTransformer
from corelib.ml.transformers.min_max import MinMaxTransformer
from corelib.ml.transformers.standar import StandardTransformer
from corelib.ml.transformers.transformer import FeatureTransformer


class TransformerType(str, enum.Enum):
    """Available feature transformers."""

    STANDARD_SCALER: TransformerType = "STANDARD_SCALER"
    MIN_MAX_SCALER: TransformerType = "MIN_MAX_SCALER"
    IDENTITY: TransformerType = "IDENTITY"


class TransformerFactory:
    """Transformer factory."""

    def __init__(self):
        """Instantiate a transformer factory."""
        self._params = {
            TransformerType.STANDARD_SCALER: {},
            TransformerType.MIN_MAX_SCALER: {},
            TransformerType.IDENTITY: {},
        }
        self._catalogue = {
            TransformerType.STANDARD_SCALER: StandardTransformer,
            TransformerType.MIN_MAX_SCALER: MinMaxTransformer,
            TransformerType.IDENTITY: IdentityTransformer,
        }

    def create(self, transformer_type: TransformerType) -> FeatureTransformer:
        """Instantiate an ML algorithm implementation.

        Args:
            transformer_type: TransformerType
                Feature transformer type.

        Returns:
            BaseEstimator:
                ML algorithm instance.
        """
        params = self._params.get(transformer_type, None)

        if params is None:
            raise NotImplementedError(
                f"{transformer_type} parameters not implemented"
            )

        transformer = self._catalogue.get(transformer_type, None)

        if transformer is None:
            raise NotImplementedError(f"{transformer_type} not implemented")

        return transformer(**params)
