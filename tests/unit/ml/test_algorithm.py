# -*- coding: utf-8 -*-
"""Unit test suit for ML algorithms.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

import pytest

from corelib.ml.algorithms.algorithm import Algorithm
from corelib.ml.algorithms.light_gbm import LightGBM


@pytest.mark.unit
def test_algorithm_abstract_class() -> None:
    """Test interface raise an error when not implemented correctly."""

    class IncompleteAlgorithm(Algorithm):
        """IncompleteAlgorithm."""

        pass

    with pytest.raises(TypeError):
        IncompleteAlgorithm({}, {})


@pytest.mark.unit
def test_complete_subclass() -> None:
    """Test complete interface implementation."""

    class CompleteAlgorithm(Algorithm):
        """CompleteAlgorithm."""

        def fit_algorithm(self, features, target, hyper_parameters):
            """Fit algorithm."""
            pass

        def get_predictions(self, features):
            """Get predictions."""
            pass

        def get_scores(self, features):
            """Get scores."""
            pass

        def get_fit_param(self):
            """Get fit params."""
            pass

    # This should not raise any exceptions
    algo = CompleteAlgorithm({}, {})
    assert isinstance(algo, CompleteAlgorithm)


@pytest.mark.unit
def test_lightgbm_fit_algorithm(algorithm_artifacts: Dict[str, Any]) -> None:
    """Test lightgbm fit algorithm."""
    algorithm = LightGBM(
        default_params=algorithm_artifacts.get("default_params"),
        hpo_params=algorithm_artifacts.get("hpo_params"),
    )
    algorithm.fit_algorithm(
        features=algorithm_artifacts.get("features"),
        target=algorithm_artifacts.get("target"),
        hyper_parameters=algorithm_artifacts.get("default_params").__dict__,
    )
    assert algorithm.gbm is not None


@pytest.mark.unit
def test_lightgbm_get_predictions_scores(
    algorithm_artifacts: Dict[str, Any]
) -> None:
    """Test lightgbm get predictions."""
    algorithm = LightGBM(
        default_params=algorithm_artifacts.get("default_params"),
        hpo_params=algorithm_artifacts.get("hpo_params"),
    )
    algorithm.fit_algorithm(
        features=algorithm_artifacts.get("features"),
        target=algorithm_artifacts.get("target"),
        hyper_parameters=algorithm_artifacts.get("default_params").__dict__,
    )

    predictions = algorithm.get_predictions(
        algorithm_artifacts.get("features")
    )
    scores = algorithm.get_scores(algorithm_artifacts.get("features"))
    assert len(predictions) == len(algorithm_artifacts.get("target"))
    assert len(scores) == len(algorithm_artifacts.get("target"))


@pytest.mark.unit
def test_get_fit_param(algorithm_artifacts: Dict[str, Any]) -> None:
    """Test ligthgbm get parameters."""
    algorithm = LightGBM(
        default_params=algorithm_artifacts.get("default_params"),
        hpo_params=algorithm_artifacts.get("hpo_params"),
    )

    fit_params = algorithm.get_fit_param()
    assert fit_params["algorithm_name"] == "LightGBM"


@pytest.mark.unit
def test_prediction_before_fit(algorithm_artifacts: Dict[str, Any]) -> None:
    """Test lightgbm raise error when predict before fitting."""
    algorithm = LightGBM(
        default_params=algorithm_artifacts.get("default_params"),
        hpo_params=algorithm_artifacts.get("hpo_params"),
    )

    with pytest.raises(AttributeError):
        algorithm.get_predictions(algorithm_artifacts.get("features"))

    with pytest.raises(AttributeError):
        algorithm.get_scores(algorithm_artifacts.get("features"))
