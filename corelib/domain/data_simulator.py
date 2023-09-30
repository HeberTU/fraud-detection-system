# -*- coding: utf-8 -*-
"""Data simulator functions.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import List

import numpy as np

from corelib.domain.models import CustomerProfile


def generate_customer_profiles_list(
    n_customers: int,
    customer_uniform_lower_bound: int,
    customer_uniform_upper_bound: int,
    amount_uniform_lower_bound: int,
    amount_uniform_upper_bound: int,
    trans_uniform_lower_bound: int,
    trans_uniform_upper_bound: int,
    random_state: int = 0,
) -> List[CustomerProfile]:
    """Generate randomly customer profiles.

     Every customer is different in their spending habits. This will be
     simulated by defining some properties for each customer. The main
     properties will be their geographical location, their spending frequency,
     and their spending amounts.

    Args:
        n_customers: int
            Number of simulated customers.
        customer_uniform_lower_bound: int
            Lower limit from the uniform distribution that will be used to
            simulate the customer geographical data.
        customer_uniform_upper_bound: int
            Upper limit from the uniform distribution that will be used to
            simulate the customer geographical data.
        amount_uniform_lower_bound: int
            Lower limit from the uniform distribution that will be used to
            simulate the customer spending amounts data.
        amount_uniform_upper_bound: int
            Upper limit from the uniform distribution that will be used to
            simulate the customer spending amounts data.
        trans_uniform_lower_bound: int
            Lower limit from the uniform distribution that will be used to
            simulate the customer spending frequency data.
        trans_uniform_upper_bound: int
            Upper limit from the uniform distribution that will be used to
            simulate the customer spending frequency data.
        random_state: int
            Random seed for reproducibility purposes.

    Returns:
        List[CustomerProfile]:
            List containing the n simulated customer profiles.
    """
    np.random.seed(random_state)

    customer_profile_list = []

    for customer_id in range(n_customers):
        mean_amount = np.random.uniform(
            low=amount_uniform_lower_bound,
            high=amount_uniform_upper_bound,
        )

        customer_profile = CustomerProfile(
            customer_id=customer_id,
            x_customer_id=np.random.uniform(
                low=customer_uniform_lower_bound,
                high=customer_uniform_upper_bound,
            ),
            y_customer_id=np.random.uniform(
                low=customer_uniform_lower_bound,
                high=customer_uniform_upper_bound,
            ),
            mean_amount=mean_amount,
            std_amount=mean_amount / 2,
            mean_nb_tx_per_day=np.random.uniform(
                low=trans_uniform_lower_bound,
                high=trans_uniform_upper_bound,
            ),
        )
        customer_profile_list.append(customer_profile)

    return customer_profile_list
