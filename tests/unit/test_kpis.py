# -*- coding: utf-8 -*-
"""Unit test suit for the domain kpis moddule.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
import pytest

from corelib.domain import kpis


@pytest.mark.unit
def test_card_precision_top_k_day(test_data: pd.DataFrame) -> None:
    """Test card_precision_top_k_day."""
    test_data = test_data[
        test_data["tx_datetime"] == pd.to_datetime("2023-10-01")
    ].copy()
    result = kpis.card_precision_top_k_day(test_data, top_k=3)

    # Check results with expected
    assert result["card_precision_top_k"] == 2 / 3
    assert result["perfect_card_precision_top_k"] == 0.02
    assert result["detected_compromised_cards_list"] == [1, 2]


@pytest.mark.unit
def test_card_precision_top_k(test_data: pd.DataFrame) -> None:
    """Test test_card_precision_top_k."""
    result = kpis.card_precision_top_k(
        test_data=test_data, top_k=3, remove_detected_compromised_cards=True
    )

    # Expected results are based on given mock data
    assert result == (0.667, 0.02)
