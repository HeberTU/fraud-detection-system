# -*- coding: utf-8 -*-
"""Unit test suit for ML algorithms.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest

from corelib.ml.algorithms.algorithm import Algorithm


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
