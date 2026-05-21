"""
Machine Learning XGBoost Executor
机器学习XGBoost执行器

单方XGBoost任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import LocalBaseModel
from .base import (
    XGBoostModel,
    XGBoostTrainer,
    XGBoostPredictor,
)

logger = logging.getLogger(__name__)


class MLXGBExecutor(LocalBaseModel):
    """
    机器学习XGBoost执行器

    执行单方XGBoost训练或预测任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 任务类型（train, predict）
        self.task_type = self.common_params.get("task_type", "train")
        # 机器学习任务类型（classification, regression）
        self.ml_task = self.common_params.get("ml_task", "classification")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 模型保存/加载路径
        self.model_path = self.common_params.get("model_path", "")

        # 模型参数
        self.n_estimators = self.common_params.get("n_estimators", 100)
        self.max_depth = self.common_params.get("max_depth", 6)
        self.learning_rate = self.common_params.get("learning_rate", 0.1)
        self.min_child_weight = self.common_params.get("min_child_weight", 1)
        self.subsample = self.common_params.get("subsample", 0.8)
        self.colsample_bytree = self.common_params.get("colsample_bytree", 0.8)
        self.reg_alpha = self.common_params.get("reg_alpha", 0.0)
        self.reg_lambda = self.common_params.get("reg_lambda", 1.0)
        self.scale_pos_weight = self.common_params.get("scale_pos_weight", 1.0)
        self.random_state = self.common_params.get("random_state", 42)

        # 训练参数
        self.test_size = self.common_params.get("test_size", 0.2)
        self.stratify = self.common_params.get("stratify", True)
        self.early_stopping_rounds = self.common_params.get("early_stopping_rounds", None)

    def run(self) -> Dict[str, Any]:
        """执行XGBoost任务"""
        logger.info(f"MLXGBExecutor: Starting {self.task_type} task")

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

        stratify_param = labels if (self.stratify and self.ml_task == 'classification') else None
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        # 创建训练器
        model_params = {
            "task": self.ml_task,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
        }

        train_params = {
            "early_stopping_rounds": self.early_stopping_rounds,
        }

        trainer = XGBoostTrainer(model_params=model_params, train_params=train_params)

        # 训练模型
        model = trainer.train(X_train, y_train, X_val, y_val)
        metrics = trainer.get_metrics()
        feature_importances = model.get_feature_importances()

        result = {
            "task_type": "train",
            "ml_task": self.ml_task,
            "data_shape": {"rows": len(data), "columns": len(data.columns)},
            "train_shape": {"rows": len(X_train), "columns": len(X_train.columns)},
            "val_shape": {"rows": len(X_val), "columns": len(X_val.columns)},
            "metrics": metrics,
            "feature_importances": feature_importances,
            "model_params": model.get_params(),
        }

        # 保存模型
        if self.model_path:
            model.save(self.model_path)
            result["model_path"] = self.model_path

        # 保存结果
        if self.output_path:
            self._save_result(result, self.output_path)

        logger.info(f"MLXGBExecutor: Training completed. Metrics: {metrics}")
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
            predictor = XGBoostPredictor(model_path=self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": f"Failed to load model: {e}"}

        # 预测
        try:
            predictions = predictor.predict(data)

            # 如果是分类任务，获取概率
            if self.ml_task == 'classification':
                try:
                    probabilities = predictor.predict_proba(data)
                except:
                    probabilities = None
            else:
                probabilities = None
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

        result = {
            "task_type": "predict",
            "ml_task": self.ml_task,
            "data_shape": {"rows": len(data), "columns": len(data.columns)},
            "predictions_shape": len(predictions),
            "feature_importances": predictor.get_feature_importances(),
        }

        # 保存预测结果
        if self.output_path:
            pred_df = data.copy()
            pred_df['prediction'] = predictions

            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    pred_df[f'probability_class_{i}'] = probabilities[:, i]

            if self.output_path.endswith('.csv'):
                pred_df.to_csv(self.output_path, index=False)
            else:
                self._save_result({
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist() if probabilities is not None else None,
                }, self.output_path)

            result["output_path"] = self.output_path

        logger.info("MLXGBExecutor: Prediction completed")
        return result
