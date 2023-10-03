# -*- coding: utf-8 -*-
"""Validation routine.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from corelib.data_schemas.data_schema import BaseSchema


def validate_and_coerce_schema(
    data: pd.DataFrame, schema_class: BaseSchema
) -> DataFrame[BaseSchema]:
    """Coerce data to data schema.

    Args:
        data: pd.DataFrame
            Data Frame to validate a coerce.
        schema_class:
            Expected Data Schema.

    Returns:
        DataFrame[BaseSchema]:
            validated an coerced data.
    """

    @pa.check_types
    def check_articles(
        data_frame: DataFrame[schema_class],
    ) -> DataFrame[schema_class]:
        return data_frame

    return check_articles(data)
