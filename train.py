# -*- coding: utf-8 -*-
"""Train script.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib.data_repositories.data_repository_factory import (
    DataRepositoryFactory,
    DataRepositoryType,
)


def main():
    """Execute main script."""
    data_repository = DataRepositoryFactory().create(
        data_repository_type=DataRepositoryType.SYNTHETIC
    )
    transactions_df = data_repository.load_data()
    processed_data = data_repository.preprocess(data=transactions_df)

    return processed_data


if __name__ == "__main__":
    main()
