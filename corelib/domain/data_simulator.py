# -*- coding: utf-8 -*-
"""Data simulator functions.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import List

import numpy as np
from numpy.typing import NDArray

from corelib.domain import models


def generate_customer_profiles_list(
    n_customers: int,
    customer_uniform_lower_bound: int,
    customer_uniform_upper_bound: int,
    amount_uniform_lower_bound: int,
    amount_uniform_upper_bound: int,
    trans_uniform_lower_bound: int,
    trans_uniform_upper_bound: int,
    random_state: int = 0,
) -> List[models.CustomerProfile]:
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

        customer_profile = models.CustomerProfile(
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


def generate_terminal_profiles_list(
    n_terminals: int,
    terminal_uniform_lower_bound: int,
    terminal_uniform_upper_bound: int,
    random_state: int = 0,
) -> List[models.TerminalProfiles]:
    """Generate randomly terminal profiles.

     Terminal properties will simply consist of a geographical location.

    Args:
        n_terminals: int
            Number of simulated terminal.
        terminal_uniform_lower_bound: int
            Lower limit from the uniform distribution that will be used to
            simulate the customer geographical data.
        terminal_uniform_upper_bound: int
            Upper limit from the uniform distribution that will be used to
            simulate the customer geographical data.
        random_state: int
            Random seed for reproducibility purposes.

    Returns:
        List[CustomerProfile]:
            List containing the n simulated customer profiles.
    """
    np.random.seed(random_state)

    terminal_profile_list = []

    for terminal_id in range(n_terminals):

        terminal_profile = models.TerminalProfiles(
            terminal_id=terminal_id,
            x_terminal_id=np.random.uniform(
                low=terminal_uniform_lower_bound,
                high=terminal_uniform_upper_bound,
            ),
            y_terminal_id=np.random.uniform(
                low=terminal_uniform_lower_bound,
                high=terminal_uniform_upper_bound,
            ),
        )
        terminal_profile_list.append(terminal_profile)

    return terminal_profile_list


def get_available_terminals_for_customer(
    x_y_customer: NDArray,
    x_y_terminals: NDArray,
    radius: float,
) -> List[int]:
    """Get the customer valid terminals.

    We will assume that customers only make transactions on terminals that are
    within a radius of r of their geographical locations.

    Args:
        x_y_customer: NDArray
            Array containing the customer location in a 100 x 100 grid.
        x_y_terminals: NDArray
            Array containing the terminal location in a 100 x 100 grid.
        radius: float
            Radius representing the maximum distance for a customer to use a
            terminal.
    """
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    available_terminals = list(np.where(dist_x_y < radius)[0])

    return available_terminals
