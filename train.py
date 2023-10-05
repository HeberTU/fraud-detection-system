# -*- coding: utf-8 -*-
"""Train script.

Created on: 30/9/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import argparse

from corelib import (
    data_repositories,
    ml,
)


def main(do_hpo: bool):
    """Execute main script."""
    estimator = ml.EstimatorFactory().create(
        estimator_type=ml.EstimatorType.ML_ESTIMATOR,
        data_repository_type=data_repositories.DataRepositoryType.SYNTHETIC,
        evaluator_type=ml.EvaluatorType.TIME_EVALUATOR,
        algorithm_type=ml.AlgorithmType.LIGHT_GBM,
        transformer_type=ml.TransformerType.MIN_MAX_SCALER,
        do_hpo=do_hpo,
    )

    scores = estimator.creat_model()

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and deploy the fraud detection model."
    )
    parser.add_argument(
        "--do-hpo",
        action="store_true",
        help="If indicated, the estimator will perform hpo routine.",
    )

    args = parser.parse_args()
    if vars(args) == {}:
        parser.print_help()
        exit(1)

    main(do_hpo=args.do_hpo)
