# -*- coding: utf-8 -*-
"""Local Repository module.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import numpy as np
import pandas as pd

from corelib import utils
from corelib.config import settings
from corelib.data_repositories.data_reposotory import DataRepository
from corelib.domain import feature_transformations


class Local(DataRepository):
    """Local Data Repository."""

    def __init__(
        self,
        file_name: str,
    ):
        """Instantiate a local data repository.

        Args:
            file_name: str
                File name with extension.
        """
        self.file_name = file_name

    @utils.timer
    def load_data(self) -> pd.DataFrame:
        """Load the credit card transactional data.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        transactions_df = pd.read_csv(
            filepath_or_buffer=settings.DATA_PATH / self.file_name,
            parse_dates=["TX_DATETIME"],
            dtype={
                "CUSTOMER_ID": np.int64,
                "SECTOR_ID": np.int64,
                "TX_FRAUD": np.int8,
            },
        )
        transactions_df = transactions_df.rename(
            columns={
                "CUSTOMER_ID": "customer_id",
                "TX_DATETIME": "tx_datetime",
                "SECTOR_ID": "sector_id",
                "TX_FRAUD": "tx_fraud",
                "TX_AMOUNT": "tx_amount",
            }
        )

        transactions_df["transaction_id"] = range(len(transactions_df))

        return transactions_df.set_index("transaction_id")

    @utils.timer
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess credit card transactional data to fit an ML algorithm.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        data = feature_transformations.get_time_since_previous_transaction(
            transactions_df=data,
            datetime_col="tx_datetime",
            grouping_column="customer_id",
        )

        data = feature_transformations.aggregate_feature(
            transactions_df=data,
            windows_size_in_days=[1, 7, 30],
            time_unit=feature_transformations.TimeUnits.DAYS,
            feature_name="tx_amount",
            agg_func_list=[
                feature_transformations.AggFunc.MEAN,
                feature_transformations.AggFunc.COUNT,
            ],
            datetime_col="tx_datetime",
            index_name="transaction_id",
            grouping_column="customer_id",
            delay_period=0,
        )

        data = feature_transformations.aggregate_feature(
            transactions_df=data,
            windows_size_in_days=[1, 7],
            time_unit=feature_transformations.TimeUnits.DAYS,
            feature_name="time_since_last_tx",
            agg_func_list=[
                feature_transformations.AggFunc.MEAN,
            ],
            datetime_col="tx_datetime",
            index_name="transaction_id",
            grouping_column="customer_id",
            delay_period=0,
        )

        data = feature_transformations.aggregate_feature(
            transactions_df=data,
            windows_size_in_days=[1, 7, 30],
            time_unit=feature_transformations.TimeUnits.DAYS,
            feature_name="tx_fraud",
            agg_func_list=[
                feature_transformations.AggFunc.SUM,
                feature_transformations.AggFunc.COUNT,
            ],
            datetime_col="tx_datetime",
            index_name="transaction_id",
            grouping_column="sector_id",
            delay_period=7,
        )

        data = feature_transformations.aggregate_feature(
            transactions_df=data,
            windows_size_in_days=[1, 7, 30],
            time_unit=feature_transformations.TimeUnits.DAYS,
            feature_name="tx_fraud",
            agg_func_list=[
                feature_transformations.AggFunc.SUM,
                feature_transformations.AggFunc.COUNT,
            ],
            datetime_col="tx_datetime",
            index_name="transaction_id",
            grouping_column="customer_id",
            delay_period=7,
        )

        data = data.assign(
            sector_id_mean_tx_fraud_1_days=(
                lambda x: (
                    x.sector_id_sum_tx_fraud_1_days
                    / x.sector_id_count_tx_fraud_1_days
                )
            ),
            sector_id_mean_tx_fraud_7_days=(
                lambda x: (
                    x.sector_id_sum_tx_fraud_7_days
                    / x.sector_id_count_tx_fraud_7_days
                )
            ),
            sector_id_mean_tx_fraud_30_days=(
                lambda x: (
                    x.sector_id_sum_tx_fraud_30_days
                    / x.sector_id_count_tx_fraud_30_days
                )
            ),
            customer_id_mean_tx_fraud_1_days=(
                lambda x: (
                    x.customer_id_sum_tx_fraud_1_days
                    / x.customer_id_count_tx_fraud_1_days
                )
            ),
            customer_id_mean_tx_fraud_7_days=(
                lambda x: (
                    x.customer_id_sum_tx_fraud_7_days
                    / x.customer_id_count_tx_fraud_7_days
                )
            ),
            customer_id_mean_tx_fraud_30_days=(
                lambda x: (
                    x.customer_id_sum_tx_fraud_30_days
                    / x.customer_id_count_tx_fraud_30_days
                )
            ),
        )

        data = data.fillna(
            value={
                "sector_id_mean_tx_fraud_1_days": 0,
                "sector_id_mean_tx_fraud_7_days": 0,
                "sector_id_mean_tx_fraud_30_days": 0,
                "customer_id_mean_tx_fraud_1_days": 0,
                "customer_id_mean_tx_fraud_7_days": 0,
                "customer_id_mean_tx_fraud_30_days": 0,
            }
        )

        return data
