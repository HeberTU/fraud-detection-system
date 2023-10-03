# -*- coding: utf-8 -*-
"""Data Schema Factory.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Dict

from corelib import data_repositories as dr
from corelib.data_schemas.data_schema import BaseSchema
from corelib.data_schemas.synthetic_schema import (
    SyntheticFeaturesSchema,
    SyntheticTargetSchema,
)


class DataSchemaFactory:
    """Data Schema factory."""

    def __init__(self):
        """Initialize data schema factory."""
        self._catalogue = {
            dr.DataRepositoryType.SYNTHETIC: {
                "feature_space": SyntheticFeaturesSchema,
                "target": SyntheticTargetSchema,
            }
        }

    def create(
        self, data_repository_type: dr.DataRepositoryType
    ) -> Dict[str, BaseSchema]:
        """Instantiate the data schemas.

        Args:
            data_repository_type: dr.DataRepositoryType
                Data repository type.

        Returns:
            Dict[str, BaseSchema]:
                Data Schemas
        """
        data_schemas = self._catalogue.get(data_repository_type, None)

        if data_schemas is None:
            raise NotImplementedError(
                f"{data_repository_type} not implemented"
            )

        return data_schemas
