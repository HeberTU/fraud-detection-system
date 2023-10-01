# -*- coding: utf-8 -*-
"""Binary encoding features.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
from pandera.typing import Series


def is_weekday(tx_datetime: Series[pd.Timestamp]) -> Series[int]:
    """Check if the provided date is weekend.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.

    Returns:
        int:
            1 if the provided date is weekend 0 otherwise.

    """
    return (tx_datetime.dt.weekday >= 5).astype(int)


def is_night(tx_datetime: Series[pd.Timestamp]) -> Series[int]:
    """Check if the provided date fall during night.

    A transaction occurs during the day (0) or during the night (1).
    The night is defined as hours that are between 0pm and 6am.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.

    Returns:
        int:
            1 if the provided date fall during the night 0 otherwise.
    """
    return (tx_datetime.dt.hour <= 6).astype(int)
