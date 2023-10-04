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

from sklearn.metrics import roc_auc_score

from corelib.ml.metrics.metric import (
    Metric,
    Results,
    TrueValues,
)


class ROCAUCScore(Metric):
    """Area Under the Receiver Operating Characteristic Curve.

    ROC curves are relevant to get a sense of a classifier performance over the
    whole range of possible False positive rates (FPR). Their interest for
    fraud detection is however limited since an important goal of fraud
    detection is to keep the FPR very low.

    For Example:

    Imagine a dataset containing 100,000 transaction per day

    Assuming that 100 transactions can be checked by the investigators every
    day, that is, around 0.1% of the transactions can be checked. Therefore,
    any FPR = FP/( TN + FP ) higher than 0.1% will raise more alerts that can
    be handled by investigators. That is, any FPR higher than 0.1% is already
    too high.

    As a result, due to the imbalanced nature of the problem, 99.9% of what is
    represented on the ROC curve has little relevance from the perspective of
    an operational fraud detection system where fraudulent transactions must be
    checked by a limited team of investigators.
    """

    name: str = "roc_auc_score"
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

        score = roc_auc_score(
            y_true=true_values.tx_fraud,
            y_score=results.scores,
            **self.params,
        )

        return score
