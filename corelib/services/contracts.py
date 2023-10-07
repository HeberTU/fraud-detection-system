# -*- coding: utf-8 -*-
"""Service Contracts.

Created on: 4/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""

from pydantic import (
    BaseModel,
    Field,
)


class PredictionRequest(BaseModel):
    """Prediction request contract."""

    tx_datetime: int = Field(..., description="Unix timestamp in milliseconds")
    tx_amount: float = Field(..., description="Transaction amount")
    customer_id_mean_tx_amount_1_days: float = Field(
        ...,
        description="Mean transaction amount for customer in the last 1 day",
    )
    customer_id_count_tx_amount_1_days: float = Field(
        ...,
        description="Count of transactions for customer in the last 1 day",
    )
    customer_id_mean_tx_amount_7_days: float = Field(
        ...,
        description="Mean transaction amount for customer in the last 7 days",
    )
    customer_id_count_tx_amount_7_days: float = Field(
        ...,
        description="Count of transactions for customer in the last 7 days",
    )
    customer_id_mean_tx_amount_30_days: float = Field(
        ...,
        description="Mean transaction amount for customer in the last 30 days",
    )
    customer_id_count_tx_amount_30_days: float = Field(
        ...,
        description="Count of transactions for customer in the last 30 days",
    )
    sector_id_mean_tx_fraud_1_days: float = Field(
        ...,
        description="Risk score sector in the last 1 day",
    )
    sector_id_mean_tx_fraud_7_days: float = Field(
        ...,
        description="Risk score sector in the last 7 days",
    )
    sector_id_mean_tx_fraud_30_days: float = Field(
        ...,
        description="Risk score sector in the last 30 days",
    )
    customer_id_mean_tx_fraud_1_days: float = Field(
        ...,
        description="Risk score for customer in the last 1 day",
    )
    customer_id_mean_tx_fraud_7_days: float = Field(
        ...,
        description="Risk score for customer in the last 7 day",
    )
    customer_id_mean_tx_fraud_30_days: float = Field(
        ...,
        description="Risk score for customer in the last 30 day",
    )
    time_since_last_tx: int = Field(
        ..., description="Time since the last transaction"
    )
    customer_id_mean_time_since_last_tx_1_days: float = Field(
        ...,
        description="Mean time since the last transaction last 1 day",
    )
    customer_id_mean_time_since_last_tx_7_days: float = Field(
        ...,
        description="Mean time since the last transaction last 7 day",
    )


class PredictionResponse(BaseModel):
    """Prediction contract."""

    transaction_id: str = Field(
        ..., description="Unique Identifier transaction."
    )
    transaction_to_block: int = Field(
        ..., description="(1 = we block the transaction, 0 = we don't)"
    )
