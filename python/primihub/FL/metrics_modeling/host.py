"""
Federated Learning Metrics Modeling Host/Guest
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.file import save_json_file
from primihub.FL.utils.dataset import read_data

from .base import FederatedMetrics, ModelPerformanceAnalyzer, FeatureImportanceAnalyzer

logger = logging.getLogger(__name__)


class MetricsModelingHost(BaseModel):
    """指标建模分析主机端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.task_type = self.common_params.get("task_type", "classification")
        self.analysis_types = self.common_params.get("analysis_types", ["metrics", "performance"])
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", -1)
        self.pred_column = self.role_params.get("pred_column", -2)

    def run(self):
        """执行指标建模分析"""
        logger.info("MetricsModelingHost: 开始指标建模分析")

        guest_parties = [p for p in self.roles if self.roles[p] == "guest"]
        guest_channel = MultiGrpcClients(
            self.node_info["self_party"],
            guest_parties,
            self.node_info,
            self.task_info,
        ) if guest_parties else None

        data = self._load_data()
        y_true = data[:, self.label_column]
        y_pred = data[:, self.pred_column]
        X = np.delete(data, [self.label_column, self.pred_column], axis=1)

        results = {}

        if "metrics" in self.analysis_types:
            metrics_calc = FederatedMetrics(
                FL_type="H", role="server", channel=guest_channel, task_type=self.task_type
            )
            results["metrics"] = metrics_calc.compute_federated(y_true, y_pred)

        if "performance" in self.analysis_types:
            analyzer = ModelPerformanceAnalyzer(task_type=self.task_type)
            results["performance"] = analyzer.analyze(y_true, y_pred)

        if "importance" in self.analysis_types:
            importance_analyzer = FeatureImportanceAnalyzer(method="correlation")
            results["feature_importance"] = importance_analyzer.analyze(X, y_true)

        if self.output_path:
            save_json_file(results, self.output_path)

        logger.info("MetricsModelingHost: 指标建模分析完成")
        return results

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])


class MetricsModelingGuest(BaseModel):
    """指标建模分析访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        self.task_type = self.common_params.get("task_type", "classification")
        self.output_path = self.common_params.get("output_path", "")

        self.data_info = self.role_params.get("data", {})
        self.label_column = self.role_params.get("label_column", -1)
        self.pred_column = self.role_params.get("pred_column", -2)

    def run(self):
        """执行指标建模分析"""
        logger.info("MetricsModelingGuest: 开始指标建模分析")

        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        host_channel = GrpcClient(
            self.node_info["self_party"],
            host_party,
            self.node_info,
            self.task_info,
        ) if host_party else None

        data = self._load_data()
        y_true = data[:, self.label_column]
        y_pred = data[:, self.pred_column]

        metrics_calc = FederatedMetrics(
            FL_type="H", role="client", channel=host_channel, task_type=self.task_type
        )
        results = metrics_calc.compute_federated(y_true, y_pred)

        if self.output_path:
            save_json_file(results, self.output_path)

        return results

    def _load_data(self) -> np.ndarray:
        if self.data_info:
            data = read_data(data_info=self.data_info, selected_column=None, droped_column=None)
            return np.array(data, dtype=np.float64)
        return np.array([])
