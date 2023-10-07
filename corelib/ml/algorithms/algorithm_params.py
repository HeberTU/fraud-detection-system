# -*- coding: utf-8 -*-
"""Algorithm default parameters.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass

import corelib.ml.hyperparam_optim as hpo


@dataclass
class DecisionTreeParams:
    """Decision tree parameters."""

    criterion: str = "gini"
    max_depth: int = 2
    random_state: int = 0


@dataclass
class LightGBMParams:
    """Light GBM HPO parameters.

    objective: str
        Name of the loss function.
    num_iterations: int:
        Number of boosting iterations.
    max_depth: int
        Limit the max depth for tree model. This is used to deal with
        over-fitting when data is small.
    num_leaves: int
        Max number of leaves in one tree
    learning_rate: float
        Shrinkage rate
    feature_fraction: float
        LightGBM will randomly select a subset of features on each
        iteration (tree) if feature_fraction is smaller than 1.0. For
        example, if you set it to 0.8, LightGBM will select 80% of
        features before training each tree.
    bagging_fraction: float
        Like feature_fraction, but this will randomly select part of
         data without resampling
    min_gain_to_split: float
        The minimal gain to perform split
    min_data_in_leaf: int
        Minimal number of data in one leaf. Can be used to deal with
        over-fitting
    lambda_l1: float
        L1 regularization
    lambda_l2: float
        L3 regularization
    """

    objective: str = "xentropy"
    num_iterations: int = 100
    max_depth: int = 3
    num_leaves: int = 30
    learning_rate: float = 0.05
    bagging_fraction: float = 1
    feature_fraction: float = 1
    min_gain_to_split: float = 1
    min_data_in_leaf: float = 1
    lambda_l1: float = 0
    lambda_l2: float = 1


@dataclass
class LightGBMHPOParams:
    """Light GBM HPO parameters.

    objective: str
        Name of the loss function.
    num_iterations: int:
        Number of boosting iterations.
    max_depth: int
        Limit the max depth for tree model. This is used to deal with
        over-fitting when data is small.
    num_leaves: int
        Max number of leaves in one tree
    learning_rate: float
        Shrinkage rate
    feature_fraction: float
        LightGBM will randomly select a subset of features on each
        iteration (tree) if feature_fraction is smaller than 1.0. For
        example, if you set it to 0.8, LightGBM will select 80% of
        features before training each tree.
    bagging_fraction: float
        Like feature_fraction, but this will randomly select part of
         data without resampling
    min_gain_to_split: float
        The minimal gain to perform split
    min_data_in_leaf: int
        Minimal number of data in one leaf. Can be used to deal with
        over-fitting
    lambda_l1: float
        L1 regularization
    lambda_l2: float
        L3 regularization
    """

    objective: hpo.CategoricalDimension = hpo.CategoricalDimension(
        categories=["xentropy"], name="objective"
    )
    num_iterations: hpo.IntegerDimension = hpo.IntegerDimension(
        interval_start=100,
        interval_end=300,
        prior=hpo.Prior.UNIFORM,
        name="num_iterations",
    )
    max_depth: hpo.IntegerDimension = hpo.IntegerDimension(
        interval_start=3,
        interval_end=5,
        prior=hpo.Prior.UNIFORM,
        name="max_depth",
    )
    num_leaves: hpo.IntegerDimension = hpo.IntegerDimension(
        interval_start=30,
        interval_end=40,
        prior=hpo.Prior.UNIFORM,
        name="num_leaves",
    )
    learning_rate: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.05,
        interval_end=0.1,
        prior=hpo.Prior.LOG_UNIFORM,
        name="learning_rate",
    )
    bagging_fraction: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.7,
        interval_end=1,
        prior=hpo.Prior.UNIFORM,
        name="bagging_fraction",
    )
    feature_fraction: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.7,
        interval_end=1,
        prior=hpo.Prior.UNIFORM,
        name="feature_fraction",
    )
    min_gain_to_split: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.7,
        interval_end=1,
        prior=hpo.Prior.UNIFORM,
        name="min_gain_to_split",
    )
    min_data_in_leaf: hpo.IntegerDimension = hpo.IntegerDimension(
        interval_start=1,
        interval_end=20,
        prior=hpo.Prior.UNIFORM,
        name="min_data_in_leaf",
    )
    lambda_l1: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.0,
        interval_end=0.99,
        prior=hpo.Prior.UNIFORM,
        name="lambda_l1",
    )
    lambda_l2: hpo.RealDimension = hpo.RealDimension(
        interval_start=0.5,
        interval_end=0.99,
        prior=hpo.Prior.UNIFORM,
        name="lambda_l2",
    )
