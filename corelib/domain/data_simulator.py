# -*- coding: utf-8 -*-
"""Data simulator functions.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import random
from typing import (
    List,
    Optional,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from corelib import utils
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


def generate_transaction(
    customer_profile: models.CustomerProfile,
    start_date: pd.Timestamp,
    day: int,
) -> Optional[models.Transaction]:
    """Generate a random transaction.

    Args:
        customer_profile: models.CustomerProfile
            Customer profile.
        start_date: pd.Timestamp
            Date from which the transactions will be generated.
        day: int
            Number of days after the start date from which the transaction will
            be simulated.

    Returns:
        Optional[models.Transaction]:
            Transaction instance in case the customer profile has valid
            terminals and the simulated hour of the daty falls on the same day.

    """
    # Time of transaction: Around noon, std 20000 seconds. This choice
    # aims at simulating the fact that most transactions occur during
    # the day.
    time_tx = int(np.random.normal(86400 / 2, 20000))

    # If transaction time between 0 and 86400 (same day), let us keep
    # it, otherwise, let us discard it.
    if (time_tx > 0) and (time_tx < 86400):

        # Amount is drawn from a normal distribution
        amount = np.random.normal(
            loc=customer_profile.mean_amount, scale=customer_profile.std_amount
        )

        # If amount negative, draw from a uniform distribution
        if amount < 0:
            amount = np.random.uniform(
                low=0, high=customer_profile.mean_amount * 2
            )

        amount = np.round(amount, decimals=2)

        if len(customer_profile.available_terminals) > 0:
            terminal_id = random.choice(
                seq=customer_profile.available_terminals
            )

            return models.Transaction(
                tx_datetime=(
                    start_date
                    + pd.Timedelta(value=day, unit="day")
                    + pd.Timedelta(value=time_tx, unit="seconds")
                ),
                customer_id=customer_profile.customer_id,
                terminal_id=terminal_id,
                tx_amount=amount,
            )

    return None


def generate_transaction_table(
    customer_profile: models.CustomerProfile,
    start_date: pd.Timestamp,
    nb_days: int,
) -> pd.DataFrame:
    """Generate transaction table.

    Args:
        customer_profile: models.CustomerProfile
            Customer profile.
        start_date: pd.Timestamp
            Date from which the transactions will be generated.
        nb_days: int
            Number of days to simulate the data.

    Returns:
        pd.DataFrame:
            Transactional data.
    """
    customer_transactions = []

    random.seed(int(customer_profile.customer_id))
    np.random.seed(int(customer_profile.customer_id))

    # For all days
    for day in range(nb_days):

        # Random number of transactions for that day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

        # If nb_tx positive, let us generate transactions
        if nb_tx <= 0:
            continue

        for tx in range(nb_tx):

            transaction = generate_transaction(
                customer_profile=customer_profile,
                start_date=start_date,
                day=day,
            )

            if transaction is None:
                continue

            customer_transactions.append(transaction)

    customer_transactions = pd.DataFrame.from_records(
        [ct.__dict__ for ct in customer_transactions]
    )

    return customer_transactions


def add_frauds(
    customer_profiles_df: pd.DataFrame,
    terminal_profiles_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    nb_days: int,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add fraudulent transactions.

    We consider the following cases:
        - Baseline fraud, anomalous amounts.
        - Phishing
        - card not present fraud.

    Args:
        customer_profiles_df: pd.DataFrame
            DataFrame containing the customer profile data.
        terminal_profiles_df: pd.DataFrame
            DataFrame containing all the terminal profiles.
        transactions_df: pd.DataFrame
            DataFrame containing all the transactions
        nb_days: int
            Number of days to simulate fraud.
        start_date: pd.Timestamp
            Date from which the transactions will be generated.

    Returns:
        pd.DataFrame
            DataFrame containing all the transactions and the simulated frauds.
    """
    # By default, all transactions are genuine
    transactions_df["tx_fraud"] = 0
    transactions_df["tx_fraud_scenario"] = 0

    # Scenario 1
    transactions_df = simulate_baseline_fraud(transactions_df=transactions_df)

    for day in range(nb_days):
        # Scenario 2
        transactions_df = simulate_phishing(
            terminal_profiles_df=terminal_profiles_df,
            transactions_df=transactions_df,
            day=day,
            start_date=start_date,
        )

        # Scenario 3
        transactions_df = simulate_card_not_present_fraud(
            customer_profiles_df=customer_profiles_df,
            transactions_df=transactions_df,
            day=day,
            start_date=start_date,
        )

    return transactions_df


def simulate_baseline_fraud(
    transactions_df: pd.DataFrame,
    amount_threshold: int = 220,
) -> pd.DataFrame:
    """Simulate a baseline fraud.

    Any transaction whose amount is more than 220 is a fraud. This scenario is
    not inspired by a real-world scenario. Rather, it will provide an obvious
    fraud pattern that should be detected by any baseline fraud detector.

    Args:
        transactions_df: pd.DataFrame
            DataFrame containing all the transactions
        amount_threshold: int:
            Any transaction with an amount larger that this threshold will be
            considered as fraud.

    Returns:
        pd.DataFrame
            DataFrame containing all the transactions and the simulated frauds.
    """
    transactions_df.loc[
        transactions_df.tx_amount > amount_threshold, "tx_fraud"
    ] = 1
    transactions_df.loc[
        transactions_df.tx_amount > amount_threshold, "tx_fraud_scenario"
    ] = 1

    return transactions_df


def simulate_phishing(
    terminal_profiles_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    day: int,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Simulate fraudulent transaction via phishing.

    Every day, a list of two terminals is drawn at random. All transactions on
    these terminals in the next 28 days will be marked as fraudulent.

    Args:
        terminal_profiles_df: pd.DataFrame
            DataFrame containing all the terminal profiles.
        transactions_df: pd.DataFrame
            DataFrame containing all the transactions
        day: int
            Number of days after the start day from which the fraudulent
            transaction are drawn.
        start_date: pd.Timestamp
            Date from which the transactions will be generated.

    Returns:
        pd.DataFrame
            DataFrame containing all the transactions and the simulated frauds.
    """
    compromised_terminals = terminal_profiles_df.terminal_id.sample(
        n=2, random_state=day
    )

    compromised_transactions = transactions_df[
        (
            transactions_df.tx_datetime
            >= start_date + pd.Timedelta(value=day, unit="days")
        )
        & (
            transactions_df.tx_datetime
            < start_date + pd.Timedelta(value=day + 28, unit="days")
        )
        & (transactions_df.terminal_id.isin(compromised_terminals))
    ]

    transactions_df.loc[compromised_transactions.index, "tx_fraud"] = 1
    transactions_df.loc[
        compromised_transactions.index, "tx_fraud_scenario"
    ] = 2

    return transactions_df


def simulate_card_not_present_fraud(
    customer_profiles_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    day: int,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Simulate where the credentials of a customer have been leaked.

    Every day, a list of 3 customers is drawn at random. In the next 14 days,
    1/3 of their transactions have their amounts multiplied by 5 and marked as
    fraudulent. The customer continues to make transactions, and transactions
    of higher values are made by the fraudster who tries to maximize their
    gains.

    Args:
        customer_profiles_df: pd.DataFrame
            DataFrame containing the customer profile data.
        transactions_df: pd.DataFrame
            DataFrame containing all the transactions
        day: int
            Number of days after the start day from which the fraudulent
            transaction are drawn.
        start_date: pd.Timestamp
            Date from which the transactions will be generated.

    Returns:
        pd.DataFrame
            DataFrame containing all the transactions and the simulated frauds.
    """
    compromised_customers = customer_profiles_df.customer_id.sample(
        n=3, random_state=day
    ).values

    compromised_transactions = transactions_df[
        (
            transactions_df.tx_datetime
            >= start_date + pd.Timedelta(value=day, unit="days")
        )
        & (
            transactions_df.tx_datetime
            < start_date + pd.Timedelta(value=day + 14, unit="days")
        )
        & (transactions_df.customer_id.isin(compromised_customers))
    ]

    nb_compromised_transactions = len(compromised_transactions)

    random.seed(day)

    index_fauds = random.sample(
        list(compromised_transactions.index.values),
        k=int(nb_compromised_transactions / 3),
    )

    transactions_df.loc[index_fauds, "tx_amount"] = (
        transactions_df.loc[index_fauds, "tx_amount"] * 5
    )
    transactions_df.loc[index_fauds, "tx_fraud"] = 1
    transactions_df.loc[index_fauds, "tx_fraud_scenario"] = 3

    return transactions_df


@utils.cacher
def simulate_credit_card_transactions_data(
    n_terminals: int,
    n_customers: int,
    geo_uniform_lower_bound: int,
    geo_uniform_upper_bound: int,
    amount_uniform_lower_bound: int,
    amount_uniform_upper_bound: int,
    trans_uniform_lower_bound: int,
    trans_uniform_upper_bound: int,
    radius: float,
    start_date: pd.Timestamp,
    nb_days: int,
    random_state: int,
) -> pd.DataFrame:
    """Simulate credit card transaction date.

    Args:
        n_terminals: int
            Number of simulated terminal.
        n_customers: int
            Number of simulated customers.
        geo_uniform_lower_bound: int
                Lower limit from the uniform distribution that will be used to
                simulate geographical data.
        geo_uniform_upper_bound: int
            Upper limit from the uniform distribution that will be used to
            simulate geographical data.
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
        start_date: pd.Timestamp
            Date from which the transactions will be generated.
        nb_days: int
            Number of day to generate data.
        radius: float
            Radius representing the maximum distance for a customer to use
            a terminal.
        random_state: int
            Random seed for reproducibility purposes.

    Returns:
        pd.DataFrame:
            DataFrame containing simulated data.
    """
    terminal_profile_list = generate_terminal_profiles_list(
        n_terminals=n_terminals,
        terminal_uniform_lower_bound=geo_uniform_lower_bound,
        terminal_uniform_upper_bound=geo_uniform_upper_bound,
        random_state=random_state,
    )

    terminal_profile_df = pd.DataFrame.from_records(
        [t.__dict__ for t in terminal_profile_list]
    )

    x_y_terminals = terminal_profile_df[
        ["x_terminal_id", "y_terminal_id"]
    ].values.astype(float)

    customer_profiles_list = generate_customer_profiles_list(
        n_customers=n_customers,
        customer_uniform_lower_bound=geo_uniform_lower_bound,
        customer_uniform_upper_bound=geo_uniform_upper_bound,
        amount_uniform_lower_bound=amount_uniform_lower_bound,
        amount_uniform_upper_bound=amount_uniform_upper_bound,
        trans_uniform_lower_bound=trans_uniform_lower_bound,
        trans_uniform_upper_bound=trans_uniform_upper_bound,
        random_state=random_state,
    )

    customer_profiles_df = pd.DataFrame.from_records(
        [c.__dict__ for c in customer_profiles_list]
    )

    customer_profiles_df["available_terminals"] = customer_profiles_df.apply(
        lambda row: get_available_terminals_for_customer(
            x_y_customer=row[["x_customer_id", "y_customer_id"]].values.astype(
                float
            ),
            x_y_terminals=x_y_terminals,
            radius=radius,
        ),
        axis=1,
    )

    transactions_df = (
        customer_profiles_df.groupby("customer_id")
        .apply(
            lambda x: generate_transaction_table(
                x.iloc[0], start_date=start_date, nb_days=nb_days
            )
        )
        .reset_index(drop=True)
    )

    transactions_df = add_frauds(
        customer_profiles_df=customer_profiles_df,
        terminal_profiles_df=terminal_profile_df,
        transactions_df=transactions_df,
        nb_days=nb_days,
        start_date=start_date,
    )

    return transactions_df
