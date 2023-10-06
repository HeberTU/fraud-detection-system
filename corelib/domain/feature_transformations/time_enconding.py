# -*- coding: utf-8 -*-
"""Time encoding features.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandera.typing import Series


class TimeEncoderFunc(str, enum.Enum):
    """Available trigonometric encoder functions."""

    SIN: TimeEncoderFunc = "SIN"
    COS: TimeEncoderFunc = "COS"
    IDENTITY: TimeEncoderFunc = "IDENTITY"


def identity(x: ArrayLike) -> Series[float]:
    """Identity function."""
    return x


def cos(x: ArrayLike) -> Series[float]:
    """Cos function."""
    return np.cos(2 * np.pi * x)


def sin(x: ArrayLike) -> Series[float]:
    """Sin function."""
    return np.sin(2 * np.pi * x)


def encode_day_of_week(
    tx_datetime: Series[pd.Timestamp], encoder_function: TimeEncoderFunc
) -> Series[float]:
    """Encode the day of the week using trigonometric func.

    To encode the cyclic nature of time we can use trigonometric functions
    (i.e., sine and cosine) to transform linear time measurements into a
    circular one.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.
        encoder_function: TimeEncoderFunc
            Function to encode time function to apply.

    Returns:
        Series[float]:
            Transformed day of the week.

    """
    _funcs = {
        TimeEncoderFunc.SIN: sin,
        TimeEncoderFunc.COS: cos,
        TimeEncoderFunc.IDENTITY: identity,
    }

    day_of_week = tx_datetime.dt.weekday

    func = _funcs.get(encoder_function)
    if func is None:
        raise NotImplementedError(f"{encoder_function} not implemented.")

    return func(day_of_week / 6)


def encode_day_time(
    tx_datetime: Series[pd.Timestamp], encoder_function: TimeEncoderFunc
) -> Series[float]:
    """Encode the time of the day using trigonometric func.

    To encode the cyclic nature of time we can use trigonometric functions
    (i.e., sine and cosine) to transform linear time measurements into a
    circular one.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.
        encoder_function: TrigonometricEncoder
            Function to encode time.

    Returns:
        Series[float]:
            Transformed day of the week.

    """
    _funcs = {
        TimeEncoderFunc.SIN: sin,
        TimeEncoderFunc.COS: cos,
        TimeEncoderFunc.IDENTITY: identity,
    }

    func = _funcs.get(encoder_function)
    if func is None:
        raise NotImplementedError(f"{encoder_function} not implemented.")

    seconds_in_day = 24 * 60 * 60
    seconds_passed = (
        tx_datetime.dt.hour * 60 * 60
        + tx_datetime.dt.minute * 60
        + tx_datetime.dt.second
    )

    return func(seconds_passed / seconds_in_day)
