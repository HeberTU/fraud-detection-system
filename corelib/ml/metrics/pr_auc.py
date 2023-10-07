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
from corelib.utils.plot import (
    plot_precision_recall_curve,
    plot_precision_recall_with_thresholds,
)


class PRAUCScore(Metric):
    """Area Under the Precision-Recall Curve."""

    name: str = "pr_auc_score"
    params: Optional[Dict[str, Any]] = None

    def measure(
        self,
        results: Results,
        true_values: TrueValues,
        plot_results: bool = False,
    ) -> float:
        """Compute Area Under the Precision-Recall Curve.

        Args:
            results: Results
                Estimator predictions.
            true_values: NDArray
                True values that we want to predict.
            plot_results: bool
                If True, model results will be plotted.

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

        if plot_results:
            plot_precision_recall_curve(
                precision=precision[::-1],
                recall=recall[::-1],
                pr_auc=score,
                pr_auc_random=true_values.tx_fraud.mean().values[0],
            )
            plot_precision_recall_with_thresholds(
                precision=precision, recall=recall, thresholds=thresholds
            )

        return score
