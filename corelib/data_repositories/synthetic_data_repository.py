# -*- coding: utf-8 -*-
"""Synthetic Data Repository.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd

from corelib import domain
from corelib.data_repositories.data_reposotory import DataRepository


class Synthetic(DataRepository):
    """Synthetic Data Repository."""

    def __init__(
        self,
        n_customers: int,
        n_terminals: int,
        geo_uniform_lower_bound: int,
        geo_uniform_upper_bound: int,
        amount_uniform_lower_bound: int,
        amount_uniform_upper_bound: int,
        trans_uniform_lower_bound: int,
        trans_uniform_upper_bound: int,
        start_date: pd.Timedelta,
        nb_days: int,
        radius: float,
        random_state: int,
    ):
        """Instantiate synthetic data repository.

        Args:
            n_customers: int
                Number of simulated customers.
            n_terminals: int
                Number of simulated terminal.
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
        """
        self.n_customers = n_customers
        self.n_terminals = n_terminals

        self.geo_uniform_lower_bound = geo_uniform_lower_bound
        self.geo_uniform_upper_bound = geo_uniform_upper_bound
        self.amount_uniform_lower_bound = amount_uniform_lower_bound
        self.amount_uniform_upper_bound = amount_uniform_upper_bound
        self.trans_uniform_lower_bound = trans_uniform_lower_bound
        self.trans_uniform_upper_bound = trans_uniform_upper_bound

        self.start_date = start_date
        self.nb_days = nb_days
        self.radius = radius

        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        """Simulate the credit card transactional data.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        terminal_profile_list = domain.generate_terminal_profiles_list(
            n_terminals=self.n_terminals,
            terminal_uniform_lower_bound=self.geo_uniform_lower_bound,
            terminal_uniform_upper_bound=self.geo_uniform_upper_bound,
            random_state=self.random_state,
        )

        terminal_profile_df = pd.DataFrame.from_records(
            [t.__dict__ for t in terminal_profile_list]
        )

        x_y_terminals = terminal_profile_df[
            ["x_terminal_id", "y_terminal_id"]
        ].values.astype(float)

        customer_profiles_list = domain.generate_customer_profiles_list(
            n_customers=self.n_customers,
            customer_uniform_lower_bound=self.geo_uniform_lower_bound,
            customer_uniform_upper_bound=self.geo_uniform_upper_bound,
            amount_uniform_lower_bound=self.amount_uniform_lower_bound,
            amount_uniform_upper_bound=self.amount_uniform_upper_bound,
            trans_uniform_lower_bound=self.trans_uniform_lower_bound,
            trans_uniform_upper_bound=self.trans_uniform_upper_bound,
            random_state=self.random_state,
        )

        customer_profiles_df = pd.DataFrame.from_records(
            [c.__dict__ for c in customer_profiles_list]
        )

        customer_profiles_df[
            "available_terminals"
        ] = customer_profiles_df.apply(
            lambda row: domain.get_available_terminals_for_customer(
                x_y_customer=row[
                    ["x_customer_id", "y_customer_id"]
                ].values.astype(float),
                x_y_terminals=x_y_terminals,
                radius=self.radius,
            ),
            axis=1,
        )

        transactions_df = (
            customer_profiles_df.groupby("customer_id")
            .apply(
                lambda x: domain.generate_transaction_table(
                    x.iloc[0], start_date=self.start_date, nb_days=self.nb_days
                )
            )
            .reset_index(drop=True)
        )

        return transactions_df

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess credit card transactional data to fit an ML algorithm.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        raise pd.DataFrame()
