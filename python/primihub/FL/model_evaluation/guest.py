"""
Model Evaluation Guest Implementation
联邦模型评估Guest端实现

Guest作为评估的参与方，提供本地预测结果。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file, load_pickle_file
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.dataset import read_data

logger = logging.getLogger(__name__)


class ModelEvaluationGuest(BaseModel):
    """
    联邦模型评估Guest端

    执行联邦模型评估的Guest角色，发送本地预测给Host。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 预测列名
        self.prediction_column = self.common_params.get("prediction_column", "pred")
        # ID列名
        self.id_column = self.common_params.get("id_column", "id")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")

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

    def run(self):
        """执行模型评估（Guest端）"""
        logger.info("ModelEvaluationGuest: Starting model evaluation")

        # 找到Host
        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        if not host_party:
            logger.error("No host party found")
            return {"error": "No host party found"}

        # 建立通信通道
        channel = GrpcClient(
            self.node_info["self_party"],
            host_party,
            self.node_info,
            self.task_info,
        )

        # 加载数据
        data = self._load_data()

        # 获取预测值
        predictions = None
        if not data.empty and self.prediction_column in data.columns:
            predictions = data[self.prediction_column].values
        else:
            # 从文件加载预测
            predictions = self._load_predictions()

        if predictions is None:
            logger.error("No predictions available")
            return {"error": "No predictions available"}

        logger.info(f"ModelEvaluationGuest: Predictions loaded, count={len(predictions)}")

        # 接收样本数量
        sample_count = channel.recv("sample_count")
        logger.info(f"ModelEvaluationGuest: Expected sample count = {sample_count}")

        # 验证预测数量
        if len(predictions) != sample_count:
            logger.warning(
                f"Prediction count mismatch: local={len(predictions)}, expected={sample_count}"
            )
            # 尝试调整
            if len(predictions) > sample_count:
                predictions = predictions[:sample_count]
            else:
                # 填充零
                padded = np.zeros(sample_count)
                padded[:len(predictions)] = predictions
                predictions = padded

        # 发送预测给Host
        channel.send("predictions", predictions.tolist())
        logger.info("ModelEvaluationGuest: Predictions sent to Host")

        # 接收评估结果
        try:
            result = channel.recv("evaluation_result")
            logger.info("ModelEvaluationGuest: Received evaluation result from Host")
        except Exception as e:
            logger.warning(f"Failed to receive evaluation result: {e}")
            result = {"status": "predictions_sent"}

        # 保存结果
        if self.output_path and isinstance(result, dict):
            if self.output_path.endswith('.json'):
                save_json_file(result, self.output_path)
            else:
                save_pickle_file(result, self.output_path)
            logger.info(f"ModelEvaluationGuest: Results saved to {self.output_path}")

        logger.info("ModelEvaluationGuest: Completed")
        return result
