"""
Model Evaluation Host Implementation
联邦模型评估Host端实现

Host作为评估的主导方，通常持有真实标签。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file, load_pickle_file
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.dataset import read_data

from .base import ClassificationEvaluator, RegressionEvaluator, SecureEvaluator

logger = logging.getLogger(__name__)


class ModelEvaluationHost(BaseModel):
    """
    联邦模型评估Host端

    执行联邦模型评估的Host角色，通常持有真实标签。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 任务类型: classification, regression
        self.task_type = self.common_params.get("task_type", "classification")
        # 评估指标列表
        self.metrics = self.common_params.get("metrics", None)
        # 标签列名
        self.label_column = self.common_params.get("label_column", "y")
        # 预测列名（如果从文件加载）
        self.prediction_column = self.common_params.get("prediction_column", "pred")
        # ID列名
        self.id_column = self.common_params.get("id_column", "id")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 是否是多分类
        self.multiclass = self.common_params.get("multiclass", False)
        # 分类阈值
        self.threshold = self.common_params.get("threshold", 0.5)
        # 评估模式: local（本地评估）, federated（联邦评估）
        self.eval_mode = self.common_params.get("eval_mode", "federated")
        # 聚合方式: sum（纵向分割预测相加）, average（取平均）
        self.aggregation = self.common_params.get("aggregation", "sum")

        # 数据信息
        self.data_info = self.role_params.get("data", {})
        # 预测数据路径
        self.prediction_path = self.role_params.get("prediction_path", "")

    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        if not self.data_info:
            logger.warning("No data info provided")
            return pd.DataFrame()

        data = read_data(data_info=self.data_info)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        return data

    def _load_predictions(self) -> Optional[np.ndarray]:
        """加载预测结果"""
        if not self.prediction_path:
            return None

        try:
            predictions = load_pickle_file(self.prediction_path)
            return np.asarray(predictions)
        except Exception as e:
            logger.warning(f"Failed to load predictions: {e}")
            return None

    def _create_evaluator(self):
        """创建评估器"""
        if self.task_type == "classification":
            return ClassificationEvaluator(
                multiclass=self.multiclass,
                threshold=self.threshold
            )
        else:
            return RegressionEvaluator()

    def run(self):
        """执行模型评估"""
        logger.info("ModelEvaluationHost: Starting model evaluation")

        if self.eval_mode == "local":
            return self._run_local_evaluation()
        else:
            return self._run_federated_evaluation()

    def _run_local_evaluation(self):
        """本地评估模式"""
        logger.info("ModelEvaluationHost: Running local evaluation")

        # 加载数据
        data = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        # 获取真实标签
        if self.label_column not in data.columns:
            logger.error(f"Label column '{self.label_column}' not found")
            return {"error": f"Label column '{self.label_column}' not found"}

        y_true = data[self.label_column].values

        # 获取预测值
        if self.prediction_column in data.columns:
            y_pred = data[self.prediction_column].values
        else:
            # 从文件加载预测
            y_pred = self._load_predictions()
            if y_pred is None:
                logger.error("No predictions available")
                return {"error": "No predictions available"}

        # 创建评估器并评估
        evaluator = self._create_evaluator()
        metrics = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            metrics=self.metrics
        )

        # 保存结果
        result = {
            "task_type": self.task_type,
            "eval_mode": "local",
            "sample_count": len(y_true),
            "metrics": metrics,
        }

        if self.output_path:
            if self.output_path.endswith('.json'):
                save_json_file(result, self.output_path)
            else:
                save_pickle_file(result, self.output_path)
            logger.info(f"ModelEvaluationHost: Results saved to {self.output_path}")

        logger.info("ModelEvaluationHost: Local evaluation completed")
        return result

    def _run_federated_evaluation(self):
        """联邦评估模式"""
        logger.info("ModelEvaluationHost: Running federated evaluation")

        # 获取Guest列表
        guest_parties = []
        for party, role in self.roles.items():
            if role == "guest":
                guest_parties.append(party)

        if not guest_parties:
            logger.warning("No guest parties found, running local evaluation")
            return self._run_local_evaluation()

        # 建立通信通道
        if len(guest_parties) > 1:
            channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )
        else:
            channel = GrpcClient(
                self.node_info["self_party"],
                guest_parties[0],
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"ModelEvaluationHost: Data loaded, shape={data.shape}")

        # 获取真实标签
        if self.label_column not in data.columns:
            logger.error(f"Label column '{self.label_column}' not found")
            return {"error": f"Label column '{self.label_column}' not found"}

        y_true = data[self.label_column].values
        sample_count = len(y_true)

        # 发送样本数量给Guest
        channel.send("sample_count", sample_count)

        # 获取Host本地预测（如果有）
        host_predictions = None
        if self.prediction_column in data.columns:
            host_predictions = data[self.prediction_column].values
        elif self.prediction_path:
            host_predictions = self._load_predictions()

        # 接收Guest的预测
        guest_predictions_list = []
        if len(guest_parties) > 1:
            for party in guest_parties:
                guest_pred = channel.recv(f"predictions_{party}")
                guest_predictions_list.append(np.asarray(guest_pred))
        else:
            guest_pred = channel.recv("predictions")
            guest_predictions_list.append(np.asarray(guest_pred))

        logger.info(f"ModelEvaluationHost: Received predictions from {len(guest_predictions_list)} guests")

        # 聚合预测
        all_predictions = []
        if host_predictions is not None:
            all_predictions.append(host_predictions)
        all_predictions.extend(guest_predictions_list)

        if self.aggregation == "sum":
            # 纵向联邦场景：各方预测相加
            aggregated_pred = np.sum(all_predictions, axis=0)
        elif self.aggregation == "average":
            # 横向联邦场景：取平均
            aggregated_pred = np.mean(all_predictions, axis=0)
        else:
            # 默认使用第一个预测（如果只有Guest的预测）
            aggregated_pred = all_predictions[0]

        # 对于分类任务，应用sigmoid或softmax
        if self.task_type == "classification" and not self.multiclass:
            # 如果是logit值，应用sigmoid
            if np.any(aggregated_pred < 0) or np.any(aggregated_pred > 1):
                aggregated_pred = 1 / (1 + np.exp(-aggregated_pred))

        # 创建评估器并评估
        evaluator = self._create_evaluator()
        metrics = evaluator.evaluate(
            y_true=y_true,
            y_pred=aggregated_pred,
            y_score=aggregated_pred if self.task_type == "classification" else None,
            metrics=self.metrics
        )

        # 准备结果
        result = {
            "task_type": self.task_type,
            "eval_mode": "federated",
            "sample_count": sample_count,
            "party_count": len(all_predictions),
            "aggregation": self.aggregation,
            "metrics": metrics,
        }

        # 发送评估结果给Guest
        channel.send("evaluation_result", result)

        # 保存结果
        if self.output_path:
            if self.output_path.endswith('.json'):
                save_json_file(result, self.output_path)
            else:
                save_pickle_file(result, self.output_path)
            logger.info(f"ModelEvaluationHost: Results saved to {self.output_path}")

        logger.info("ModelEvaluationHost: Federated evaluation completed")
        return result
