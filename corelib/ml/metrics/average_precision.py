# -*- coding: utf-8 -*-
"""Average precision.

AP summarizes a precision-recall curve as the weighted mean of precisions
achieved at each threshold, with the increase in recall from the previous
threshold used as the weight.

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
from sklearn.metrics import average_precision_score

from corelib.ml.metrics.metric import (
    Metric,
    Results,
)


class AveragePrecisionScore(Metric):
    """Area Under the Receiver Operating Characteristic Curve."""

    name: str = "average_precision_score"
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

        score = average_precision_score(
            y_true=true_values,
            y_score=results.scores,
            **self.params,
        )

        return score
