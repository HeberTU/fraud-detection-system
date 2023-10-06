# -*- coding: utf-8 -*-
"""Local Data Schema.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandera as pa
from pandera.typing import (
    DateTime,
    Float,
    Int,
    Series,
)

from corelib.data_schemas.data_schema import BaseSchema


class LocalFeaturesSchema(BaseSchema):
    """Local feature space schema."""

    tx_amount: Series[Float] = pa.Field(nullable=False)
    is_weekday: Series[Int] = pa.Field(nullable=False)
    is_night: Series[Int] = pa.Field(nullable=False)
    customer_id_mean_tx_amount_1_days: Series[Float] = pa.Field(nullable=False)
    customer_id_count_tx_amount_1_days: Series[Float] = pa.Field(
        nullable=False
    )
    customer_id_mean_tx_amount_7_days: Series[Float] = pa.Field(nullable=False)
    customer_id_count_tx_amount_7_days: Series[Float] = pa.Field(
        nullable=False
    )
    customer_id_mean_tx_amount_30_days: Series[Float] = pa.Field(
        nullable=False
    )
    customer_id_count_tx_amount_30_days: Series[Float] = pa.Field(
        nullable=False
    )
    sector_id_mean_tx_fraud_1_days: Series[Float] = pa.Field(nullable=False)
    sector_id_mean_tx_fraud_7_days: Series[Float] = pa.Field(nullable=False)
    sector_id_mean_tx_fraud_30_days: Series[Float] = pa.Field(nullable=False)


class LocalTargetSchema(BaseSchema):
    """Local target schema."""

    tx_fraud: Series[Int] = pa.Field(nullable=False)


class LocalTimeStampSchema(BaseSchema):
    """Local TimeStam schema."""

    tx_datetime: Series[DateTime] = pa.Field(nullable=False)


class LocalCustomerIDSchema(BaseSchema):
    """Synthetic TimeStam schema."""

    customer_id: Series[Int] = pa.Field(nullable=False)
