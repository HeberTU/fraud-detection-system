# -*- coding: utf-8 -*-
"""Train script.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from corelib import (
    data_repositories,
    ml,
)


def main():
    """Execute main script."""
    estimator = ml.EstimatorFactory.create(
        data_repository_type=data_repositories.DataRepositoryType.SYNTHETIC,
        evaluator_type=ml.EvaluatorType.TIME_EVALUATOR,
        algorithm_type=ml.AlgorithmType.LIGHT_GBM,
        transformer_type=ml.TransformerType.MIN_MAX_SCALER,
    )

    scores = estimator.creat_model()

    return scores


if __name__ == "__main__":
    main()
