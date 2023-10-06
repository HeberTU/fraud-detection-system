# -*- coding: utf-8 -*-
"""Chain Transformer.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Union,
)

import pandas as pd

from corelib.data_repositories.data_repository_factory import (
    DataRepositoryType,  # fmt: skip
)
from corelib.ml.transformers.day_linear_transformer import DayLinearTransformer
from corelib.ml.transformers.time_cos_transformer import TimeCosTransformer
from corelib.ml.transformers.time_sin_transformer import TimeSinTransformer
from corelib.ml.transformers.transformer import FeatureTransformer


@dataclass
class TransformedFeature:
    """Transformed feature data structure."""

    in_feature_name: str
    out_feature_name: str
    transformer: FeatureTransformer


class TransformerChainFactory:
    """Transformer chain factory."""

    def __init__(self):
        """Instantiate transformer chan factory."""
        self._transformations = {
            DataRepositoryType.LOCAL: [
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_day_linear",
                    transformer=DayLinearTransformer(),
                ),
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_time_cos",
                    transformer=TimeCosTransformer(),
                ),
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_time_sin",
                    transformer=TimeSinTransformer(),
                ),
            ],
            DataRepositoryType.SYNTHETIC: [
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_day_linear",
                    transformer=DayLinearTransformer(),
                ),
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_time_cos",
                    transformer=TimeCosTransformer(),
                ),
                TransformedFeature(
                    in_feature_name="tx_datetime",
                    out_feature_name="tx_time_sin",
                    transformer=TimeSinTransformer(),
                ),
            ],
        }

    def create(
        self, data_repository_type: DataRepositoryType
    ) -> TransformerChain:
        """Instantiate a TransformerChain.

        Args:
            data_repository_type: DataRepositoryType
                data repository type.

        Returns:
            TransformerChain:
                Instance of transformer chain.
        """
        transformed_feature_list = self._transformations.get(
            data_repository_type
        )

        if transformed_feature_list is None:
            raise NotImplementedError(
                f"TransformerChain for {data_repository_type} not implemented"
            )

        return TransformerChain(
            transformed_feature_list=transformed_feature_list
        )


class TransformerChain:
    """Transformer Chain."""

    def __init__(self, transformed_feature_list: List[TransformedFeature]):
        """Instantiate a transformer chain.

        Args:
            transformed_feature_list: List[TransformedFeature]
                List of features to transform.
        """
        self.transformed_feature_list = transformed_feature_list

    def fit(self, features: Union[pd.DataFrame, pd.Series]) -> None:
        """Fit transformations.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                Features used to fit transformations.

        Returns:
            None.
        """
        for transformed_feature in self.transformed_feature_list:

            transformed_feature.transformer.fit_transformation(
                features=features[transformed_feature.in_feature_name]
            )

    def transform(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Transform the provided features.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                features to transform.

        Returns:
            pd.DataFrame:
                Transformed data frame.
        """
        for transformed_feature in self.transformed_feature_list:
            features[
                transformed_feature.out_feature_name
            ] = transformed_feature.transformer.apply_transformation(
                features=features[transformed_feature.in_feature_name]
            )

        return features

    def fit_transform(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Fit and transform features.

        Args:
            features: Union[pd.DataFrame, pd.Series]
                features that will be used to fit transformer and that will be
                transformed.

        Returns:
            pd.DataFrame:
                Transformed data frame.
        """
        self.fit(features=features)

        return self.transform(features=features)
