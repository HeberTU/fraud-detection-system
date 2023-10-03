# -*- coding: utf-8 -*-
"""Time evaluator implementation.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    List,
    Tuple,
)

import pandas as pd

from corelib.ml import metrics
from corelib.ml.evaluators.evaluator import Evaluator


class TimeEvaluator(Evaluator):
    """Machine learning model time evaluator."""

    def __init__(
        self,
        metric_type_list: List[metrics.MetricType],
        date_colum_name: str,
        delta_test_in_days: int,
        delta_delay_in_days: int,
    ):
        """Instantiate a time evaluator class.

        Args:
            metric_type_list: List[metrics.MetricType]
                list of model metrics.
            date_colum_name: str
                Column name that contains the datetime index.
            delta_test_in_days: int
                Number of days to include in the test set.
            delta_delay_in_days: int
                Feedback delay. It accounts for the fact that, in a real-world
                fraud detection system, the label of a transaction (fraudulent
                or genuine) is only known after a customer complaint, or thanks
                to the result of a fraud investigation.
        """
        super().__init__(metric_type_list=metric_type_list)
        self.date_colum_name = date_colum_name
        self.delta_test_in_days = delta_test_in_days
        self.delta_delay_in_days = delta_delay_in_days

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data for training and testig.

        Args:
            data: pd.DataFrame
                Data to split in training and testing.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Training and testing data.
        """
        end_date = data[self.date_colum_name].max().ceil(freq="D")

        test_start_date = end_date - pd.Timedelta(
            value=self.delta_test_in_days, unit="D"
        )

        train_end_date = test_start_date - pd.Timedelta(
            value=self.delta_delay_in_days, unit="D"
        )

        test_data = data[data[self.date_colum_name] > test_start_date].copy()

        train_data = data[data[self.date_colum_name] <= train_end_date]

        return train_data, test_data
