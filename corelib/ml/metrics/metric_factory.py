# -*- coding: utf-8 -*-
"""Metrics Factory.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

from corelib.ml.metrics.average_precision import AveragePrecisionScore
from corelib.ml.metrics.card_precision_top_k import CardPrecisionTopK
from corelib.ml.metrics.metric import Metric
from corelib.ml.metrics.perf_card_precision_top_k import PerfCardPrecisionTopK
from corelib.ml.metrics.roc_auc import ROCAUCScore


class MetricType(str, enum.Enum):
    """Available metrics."""

    ROC_AUC: MetricType = "ROC_AUC"
    AVERAGE_PRECISION: MetricType = "AVERAGE_PRECISION"
    CARD_PRECISION_TOP_K: MetricType = "CARD_PRECISION_TOP_K"
    PERFECT_CARD_PRECISION_TOP_K: MetricType = "PERFECT_CARD_PRECISION_TOP_K"


class MetricFactory:
    """Metric Factory."""

    def __init__(self):
        """Initialize metric factory."""
        self._catalogue = {
            MetricType.AVERAGE_PRECISION: AveragePrecisionScore,
            MetricType.ROC_AUC: ROCAUCScore,
            MetricType.CARD_PRECISION_TOP_K: CardPrecisionTopK,
            MetricType.PERFECT_CARD_PRECISION_TOP_K: PerfCardPrecisionTopK,
        }

    def create(self, metric_type: MetricType) -> Metric:
        """Instantiate a metric implementation.

        Args:
            metric_type: MetricType
                Metric  type to instantiate.

        Returns:
            Metric:
                Metric instance.
        """
        metric = self._catalogue.get(metric_type, None)

        if metric is None:
            raise NotImplementedError(f"{metric_type} not implemented")

        return metric(name=metric.name, params=metric.params)
