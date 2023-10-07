# -*- coding: utf-8 -*-
"""Area Under the Precision-Recall Curve for a random classifier.

Created on: 7/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
    Optional,
)

from corelib.ml.metrics.metric import (
    Metric,
    Results,
    TrueValues,
)


class RandomPRAUCScore(Metric):
    """Area Under the Precision-Recall Curve for a random classifier."""

    name: str = "random_pr_auc_score"
    params: Optional[Dict[str, Any]] = None

    def measure(
        self,
        results: Results,
        true_values: TrueValues,
        plot_results: bool = False,
    ) -> float:
        """Compute PR-AUC for a random classifier.

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

        return true_values.tx_fraud.mean().values[0]
