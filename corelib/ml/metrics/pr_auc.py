# -*- coding: utf-8 -*-
"""Precision-Recall Area Under the Curve.

Created on: 7/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
    Optional,
)

from sklearn.metrics import (
    auc,
    precision_recall_curve,
)

from corelib.ml.metrics.metric import (
    Metric,
    Results,
    TrueValues,
)


class PRAUCScore(Metric):
    """Area Under the Precision-Recall Curve."""

    name: str = "pr_auc_score"
    params: Optional[Dict[str, Any]] = None

    def measure(self, results: Results, true_values: TrueValues) -> float:
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

        precision, recall, thresholds = precision_recall_curve(
            y_true=true_values.tx_fraud,
            probas_pred=results.scores,
            **self.params,
        )

        score = auc(x=recall, y=precision)

        return score
