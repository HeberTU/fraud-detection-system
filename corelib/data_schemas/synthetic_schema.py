# -*- coding: utf-8 -*-
"""Synthetic Data Schema.

Created on: 3/10/23
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


class SyntheticFeaturesSchema(BaseSchema):
    """Synthetic feature space schema."""

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
    terminal_id_mean_tx_fraud_1_days: Series[Float] = pa.Field(nullable=False)
    terminal_id_mean_tx_fraud_7_days: Series[Float] = pa.Field(nullable=False)
    terminal_id_mean_tx_fraud_30_days: Series[Float] = pa.Field(nullable=False)


class SyntheticTargetSchema(BaseSchema):
    """Synthetic target schema."""

    tx_fraud: Series[Int] = pa.Field(nullable=False)


class SyntheticTimeStampSchema(BaseSchema):
    """Synthetic TimeStam schema."""

    tx_datetime: Series[DateTime] = pa.Field(nullable=False)


class SyntheticCustomerIDSchema(BaseSchema):
    """Synthetic TimeStam schema."""

    customer_id: Series[Int] = pa.Field(nullable=False)
