# -*- coding: utf-8 -*-
"""Data repository Factory.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

from corelib.data_repositories.data_repository_params import SyntheticParams
from corelib.data_repositories.data_reposotory import DataRepository
from corelib.data_repositories.synthetic_data_repository import Synthetic


class DataRepositoryType(str, enum.Enum):
    """Available Data Repositories."""

    SYNTHETIC: DataRepositoryType = "SYNTHETIC"


class DataRepositoryFactory:
    """Data repository Factory."""

    def __init__(self):
        """Initialize data repository factory."""
        self._params = {DataRepositoryType.SYNTHETIC: SyntheticParams}
        self._catalogue = {DataRepositoryType.SYNTHETIC: Synthetic}

    def create(
        self, data_repository_type: DataRepositoryType
    ) -> DataRepository:
        """Instantiate a data repository implementation.

        Args:
            data_repository_type: DataRepositoryType
                Data repository type to instantiate.

        Returns:
            DataRepository:
                Data repository instance.
        """
        params = self._params.get(data_repository_type, None)

        if params is None:
            raise NotImplementedError(
                f"{data_repository_type} parameters not implemented"
            )

        data_repository = self._catalogue.get(data_repository_type, None)

        if data_repository is None:
            raise NotImplementedError(
                f"{data_repository_type} not implemented"
            )

        return data_repository(**params().__dict__)
