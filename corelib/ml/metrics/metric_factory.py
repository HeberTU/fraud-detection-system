# -*- coding: utf-8 -*-
"""Metrics Factory.

Created on: 2/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum

from corelib.ml.metrics.average_precision import AveragePrecisionScore
from corelib.ml.metrics.metric import Metric
from corelib.ml.metrics.roc_auc import ROCAUCScore


class MetricType(str, enum.Enum):
    """Available metrics."""

    ROC_AUC: MetricType = "ROC_AUC"
    AVERAGE_PRECISION: MetricType = "AVERAGE_PRECISION"


class MetricFactory:
    """Metric Factory."""

    def __init__(self):
        """Initialize metric factory."""
        self._catalogue = {
            MetricType.AVERAGE_PRECISION: AveragePrecisionScore,
            MetricType.ROC_AUC: ROCAUCScore,
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

        return metric()
