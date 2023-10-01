# -*- coding: utf-8 -*-
"""Data Repository Abstraction.

Created on: 29/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)

import pandas as pd


class DataRepository(ABC):
    """Data Repository.

    This class is used as an interface to retrieve the data needed to train
    the estimator.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the credit card transactional data.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess credit card transactional data to fit an ML algorithm.

        Returns:
            pd.DataFrame: Credit card transactional data.
        """
        raise NotImplementedError
