# -*- coding: utf-8 -*-
"""Precision top-k metric.

Created on: 3/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
    Optional,
)

import pandas as pd

from corelib.domain import kpis
from corelib.ml.metrics.metric import (
    Metric,
    Results,
    TrueValues,
)


class CardPrecisionTopK(Metric):
    """Card Precision top-k."""

    name: str = "Card_precision_top_k"
    params: Optional[Dict[str, Any]] = {"top_k": 100}

    def measure(
        self,
        results: Results,
        true_values: TrueValues,
        plot_results: bool = False,
    ) -> float:
        """Compute the card  precision top-k.

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
        data = pd.concat(
            objs=[
                true_values.tx_datetime,
                true_values.customer_id,
                true_values.tx_fraud,
            ],
            axis=1,
        )

        data["scores"] = results.scores

        mean_precision_top_k, _ = kpis.card_precision_top_k(
            test_data=data, top_k=self.params.get("top_k")
        )

        return mean_precision_top_k
