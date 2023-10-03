# -*- coding: utf-8 -*-
"""Data Schema Abstraction.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import List

import pandas as pd
import pandera as pa


class BaseSchema(pa.SchemaModel):
    """Base schema."""

    class Config:
        """Base Schema configurations."""

        strict = "filter"
        coerce = True

    @pa.dataframe_check
    @classmethod
    def dataframe_is_not_empty(cls, df: pd.DataFrame) -> bool:
        """Validate that we have gathered data."""
        return df.shape[0] > 0

    @classmethod
    def get_schema_columns(cls) -> List[str]:
        """Get schema columns."""
        return [col for col in cls.to_schema().columns.keys()]
