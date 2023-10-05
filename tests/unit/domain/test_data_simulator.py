# -*- coding: utf-8 -*-
"""Unit test suit for data simulator module.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import numpy as np
import pandas as pd
import pytest

from corelib import domain


@pytest.mark.unit
def test_generate_customer_profiles_list() -> None:
    """Test generate_customer_profiles_list."""
    customer_profile_list = domain.generate_customer_profiles_list(
        n_customers=3,
        customer_uniform_lower_bound=0,
        customer_uniform_upper_bound=100,
        amount_uniform_lower_bound=5,
        amount_uniform_upper_bound=50,
        trans_uniform_lower_bound=0,
        trans_uniform_upper_bound=4,
    )

    assert len(customer_profile_list) == 3
    for customer_profile in customer_profile_list:
        assert isinstance(customer_profile, domain.CustomerProfile)

        assert 0 <= customer_profile.x_customer_id <= 100
        assert 0 <= customer_profile.y_customer_id <= 100

        assert 0 <= customer_profile.mean_amount <= 50
        assert customer_profile.std_amount == customer_profile.mean_amount / 2

        assert 0 <= customer_profile.mean_nb_tx_per_day <= 4


@pytest.mark.unit
def test_generate_terminal_profiles_list() -> None:
    """Test generate_terminal_profiles_list."""
    terminal_profile_list = domain.generate_terminal_profiles_list(
        n_terminals=3,
        terminal_uniform_lower_bound=0,
        terminal_uniform_upper_bound=100,
    )

    assert len(terminal_profile_list) == 3
    for terminal_profile in terminal_profile_list:
        assert isinstance(terminal_profile, domain.TerminalProfiles)

        assert 0 <= terminal_profile.x_terminal_id <= 100
        assert 0 <= terminal_profile.y_terminal_id <= 100


@pytest.mark.unit
def test_get_available_terminals_for_customer() -> None:
    """Test get_available_terminals_for_customer."""
    terminal_profile_list = domain.generate_terminal_profiles_list(
        n_terminals=5,
        terminal_uniform_lower_bound=0,
        terminal_uniform_upper_bound=100,
        random_state=0,
    )

    terminal_profile_df = pd.DataFrame.from_records(
        [t.__dict__ for t in terminal_profile_list]
    )

    x_y_terminals = terminal_profile_df[
        ["x_terminal_id", "y_terminal_id"]
    ].values.astype(float)

    available_terminals = domain.get_available_terminals_for_customer(
        x_y_customer=np.array([83.26198455, 77.81567509]),
        x_y_terminals=x_y_terminals,
        radius=35,
    )

    assert available_terminals == [0, 1]


@pytest.mark.unit
def test_generate_transactions() -> None:
    """Test generate_transactions."""
    customer_profile = domain.CustomerProfile(
        customer_id=0,
        x_customer_id=71.5189,
        y_customer_id=60.2763,
        mean_amount=29.6966,
        std_amount=14.8483,
        mean_nb_tx_per_day=2.17953,
        available_terminals=[2, 4],
    )

    transaction = domain.generate_transaction(
        customer_profile=customer_profile,
        start_date=pd.Timestamp("2023-09-30"),
        day=1,
    )

    assert isinstance(transaction, domain.Transaction)
    assert transaction.tx_amount > 0
    assert transaction.terminal_id in customer_profile.available_terminals


@pytest.mark.unit
def test_generate_transactions_no_terminals() -> None:
    """Test generate_transactions_table."""
    customer_profile = domain.CustomerProfile(
        customer_id=0,
        x_customer_id=71.5189,
        y_customer_id=60.2763,
        mean_amount=29.6966,
        std_amount=14.8483,
        mean_nb_tx_per_day=2.17953,
    )

    transaction = domain.generate_transaction(
        customer_profile=customer_profile,
        start_date=pd.Timestamp("2023-09-30"),
        day=1,
    )

    assert transaction is None
