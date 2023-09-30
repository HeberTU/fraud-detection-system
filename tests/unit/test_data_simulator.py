# -*- coding: utf-8 -*-
"""Unit test suit for data simulator module.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
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
