# -*- coding: utf-8 -*-
"""Area Under the Receiver Operating Characteristic Curve.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
    Optional,
)

from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from corelib.ml.metrics.metric import (
    Metric,
    Results,
)


class ROCAUCScore(Metric):
    """Area Under the Receiver Operating Characteristic Curve."""

    name: str = "roc_auc_score"
    params: Optional[Dict[str, Any]] = None

    def measure(self, results: Results, true_values: NDArray) -> float:
        """Compute Area Under the Receiver Operating Characteristic Curve.

        Args:
            results: Results
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.

        Returns:
            float:
                model performance score.
        """
        if not self.params:
            self.params = {}

        score = roc_auc_score(
            y_true=true_values,
            y_score=results.scores,
            **self.params,
        )

        return score
