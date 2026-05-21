"""
Machine Learning LR Executor
机器学习逻辑回归执行器

单方逻辑回归任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import LocalBaseModel
from .base import (
    LogisticRegressionModel,
    LogisticRegressionTrainer,
    LogisticRegressionPredictor,
)

logger = logging.getLogger(__name__)


class MLLRExecutor(LocalBaseModel):
    """
    机器学习逻辑回归执行器

    执行单方逻辑回归训练或预测任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 任务类型（train, predict）
        self.task_type = self.common_params.get("task_type", "train")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型保存/加载路径
        self.model_path = self.common_params.get("model_path", "")

        # 模型参数
        self.penalty = self.common_params.get("penalty", "l2")
        self.C = self.common_params.get("C", 1.0)
        self.solver = self.common_params.get("solver", "lbfgs")
        self.max_iter = self.common_params.get("max_iter", 100)
        self.multi_class = self.common_params.get("multi_class", "auto")
        self.class_weight = self.common_params.get("class_weight", None)
        self.random_state = self.common_params.get("random_state", 42)

        # 训练参数
        self.test_size = self.common_params.get("test_size", 0.2)
        self.stratify = self.common_params.get("stratify", True)

        # 预测参数
        self.threshold = self.common_params.get("threshold", 0.5)

    def run(self) -> Dict[str, Any]:
        """执行逻辑回归任务"""
        logger.info(f"MLLRExecutor: Starting {self.task_type} task")

        if self.task_type == "train":
            return self._run_train()
        elif self.task_type == "predict":
            return self._run_predict()
        else:
            logger.error(f"Unknown task type: {self.task_type}")
            return {"error": f"Unknown task type: {self.task_type}"}

    def _run_train(self) -> Dict[str, Any]:
        """执行训练任务"""
        # 加载数据
        data, labels = self._load_data()
        if data.empty or labels is None:
            logger.error("Training requires both features and labels")
            return {"error": "No data or labels loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 分割数据
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("sklearn is required for training")

        stratify_param = labels if self.stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        # 创建训练器
        model_params = {
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "multi_class": self.multi_class,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

        trainer = LogisticRegressionTrainer(model_params=model_params)

        # 训练模型
        model = trainer.train(X_train, y_train, X_val, y_val)
        metrics = trainer.get_metrics()

        result = {
            "task_type": "train",
            "data_shape": {"rows": len(data), "columns": len(data.columns)},
            "train_shape": {"rows": len(X_train), "columns": len(X_train.columns)},
            "val_shape": {"rows": len(X_val), "columns": len(X_val.columns)},
            "metrics": metrics,
            "model_params": model.get_params(),
        }

        # 保存模型
        if self.model_path:
            model.save(self.model_path)
            result["model_path"] = self.model_path

        # 保存结果
        if self.output_path:
            self._save_result(result, self.output_path)

        logger.info(f"MLLRExecutor: Training completed. Metrics: {metrics}")
        return result

    def _run_predict(self) -> Dict[str, Any]:
        """执行预测任务"""
        # 加载数据
        data, _ = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 加载模型
        if not self.model_path:
            logger.error("Model path is required for prediction")
            return {"error": "Model path required"}

        try:
            predictor = LogisticRegressionPredictor(model_path=self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": f"Failed to load model: {e}"}

        # 预测
        try:
            predictions = predictor.predict(data)
            probabilities = predictor.predict_proba(data)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

        result = {
            "task_type": "predict",
            "data_shape": {"rows": len(data), "columns": len(data.columns)},
            "predictions_shape": len(predictions),
        }

        # 保存预测结果
        if self.output_path:
            pred_df = data.copy()
            pred_df['prediction'] = predictions
            for i in range(probabilities.shape[1]):
                pred_df[f'probability_class_{i}'] = probabilities[:, i]

            if self.output_path.endswith('.csv'):
                pred_df.to_csv(self.output_path, index=False)
            else:
                self._save_result({
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                }, self.output_path)

            result["output_path"] = self.output_path

        logger.info("MLLRExecutor: Prediction completed")
        return result
