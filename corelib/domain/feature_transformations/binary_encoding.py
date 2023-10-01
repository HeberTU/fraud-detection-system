# -*- coding: utf-8 -*-
"""Binary encoding features.

Created on: 1/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd


def is_weekday(tx_datetime: pd.Timestamp) -> int:
    """Check if the provided date is weekend.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.

    Returns:
        int:
            1 if the provided date is weekend 0 otherwise.

    """
    return int(tx_datetime.weekday() >= 5)


def is_night(tx_datetime: pd.Timestamp) -> int:
    """Check if the provided date fall during night.

    Args:
        tx_datetime: pd.Timestamp
            date to check condition on.

    Returns:
        int:
            1 if the provided date fall during the night 0 otherwise.
    """
    return int(20 <= tx_datetime.hour <= 6)
