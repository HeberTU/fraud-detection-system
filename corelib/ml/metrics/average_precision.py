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

from sklearn.metrics import average_precision_score

from corelib.ml.metrics.metric import (
    Metric,
    Results,
    TrueValues,
)


class AveragePrecisionScore(Metric):
    """Average precision.

    The Precision-Recall (PR) curve and the Average Precision (as the metric
    that computes the area under the PR curve) are better suited for fraud
    detection problems than the ROC curve and the AUC ROC, respectively.

    A classifier that classifies all examples as positive (recall of 1) has a
    precision of P /(P + N).

    This property makes the Average Precision more interesting than the AUC ROC
    in a fraud detection problem, since it better reflects the challenge
    related to the class imbalance problem.

    The Average Precision of a random classifier decreases as the class
    imbalance ratio increases.

    precision: Pr[Is Fraud | Predict Fraud]: % actual frauds from the predicted
        fraud & detected / total predicted frauds
    recall: Pr[Detect Fraud | Fraud]: % of detected frauds
        fraud & detected / total frauds

    """

    name: str = "average_precision_score"
    params: Optional[Dict[str, Any]] = None

    def measure(
        self,
        results: Results,
        true_values: TrueValues,
        plot_results: bool = False,
    ) -> float:
        """Compute Area Under the Receiver Operating Characteristic Curve.

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

        score = average_precision_score(
            y_true=true_values.tx_fraud,
            y_score=results.scores,
            **self.params,
        )

        return score
