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
    """Input features contract."""

    tx_amount: float = Field(..., description="Transaction amount")
    is_weekday: int = Field(..., description="Is it a weekday?")
    is_night: int = Field(..., description="Is it nighttime?")
    customer_id_mean_tx_amount_1_days: float = Field(
        ...,
        description="Mean transaction amount customer over the past 1 day",
    )
    customer_id_count_tx_amount_1_days: float = Field(
        ...,
        description="Transaction count for customer ID over the past 1 day",
    )
    customer_id_mean_tx_amount_7_days: float = Field(
        ...,
        description="Mean transaction amount customer over the past 7 days",
    )
    customer_id_count_tx_amount_7_days: float = Field(
        ...,
        description="Transaction count for customer ID over the past 7 days",
    )
    customer_id_mean_tx_amount_30_days: float = Field(
        ...,
        description="Mean transaction amount customer over the past 30 days",
    )
    customer_id_count_tx_amount_30_days: float = Field(
        ...,
        description="Transaction count for customer ID over the past 30 days",
    )
    terminal_id_mean_tx_fraud_1_days: float = Field(
        ...,
        description="Mean fraud transaction for terminal over the past 1 day",
    )
    terminal_id_mean_tx_fraud_7_days: float = Field(
        ...,
        description="Mean fraud transaction for terminal over the past 7 days",
    )
    terminal_id_mean_tx_fraud_30_days: float = Field(
        ...,
        description="Mean fraud transaction terminal over the past 30 days",
    )


class PredictionResponse(BaseModel):
    """Prediction contract."""

    prediction: float = Field(
        ..., description="Predicted value from the ML model"
    )
