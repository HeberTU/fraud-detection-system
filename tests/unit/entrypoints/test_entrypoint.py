# -*- coding: utf-8 -*-
"""Unit testing for the entrypoint.

Created on: 5/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest

from corelib.entrypoints.routes import get_estimator
from corelib.ml.estimators.estimator import Estimator


@pytest.mark.unit
def test_initialization() -> None:
    """Test the estimator singleton is correctly instantiated."""
    assert isinstance(get_estimator(), Estimator)
