# -*- coding: utf-8 -*-
"""Perfect cards precision top-k.

Created on: 4/10/23
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


class PerfCardPrecisionTopK(Metric):
    """Perfect Card Precision top-k."""

    name: str = "Perfect_card_precision_top_k"
    params: Optional[Dict[str, Any]] = {"top_k": 100}

    def measure(
        self,
        results: Results,
        true_values: TrueValues,
        plot_results: bool = False,
    ) -> float:
        """Compute the perfect card precision top-k.

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

        _, perfect_card_precision_top_k = kpis.card_precision_top_k(
            test_data=data, top_k=self.params.get("top_k")
        )

        return perfect_card_precision_top_k
