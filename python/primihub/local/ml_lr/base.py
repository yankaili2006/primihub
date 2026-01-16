"""
Logistic Regression Base Classes
逻辑回归基础类

提供逻辑回归模型的训练和预测功能。
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """
    逻辑回归模型

    支持二分类和多分类。
    """

    def __init__(self,
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 100,
                 multi_class: str = 'auto',
                 class_weight: Optional[Union[str, Dict]] = None,
                 random_state: int = 42,
                 verbose: int = 0):
        """
        初始化逻辑回归模型

        Args:
            penalty: 正则化类型（l1, l2, elasticnet, none）
            C: 正则化强度的倒数
            solver: 优化算法
            max_iter: 最大迭代次数
            multi_class: 多分类策略（auto, ovr, multinomial）
            class_weight: 类别权重
            random_state: 随机种子
            verbose: 日志详细程度
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self._is_fitted = False
        self._classes = None
        self._feature_names = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'LogisticRegressionModel':
        """
        训练逻辑回归模型

        Args:
            X: 特征数据
            y: 标签数据

        Returns:
            self
        """
        logger.info("Training LogisticRegression model")

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("sklearn is required for LogisticRegressionModel")

        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 创建并训练模型
        self._model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self._model.fit(X, y)
        self._classes = self._model.classes_
        self._is_fitted = True

        logger.info(f"Model trained successfully. Classes: {self._classes}")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征数据

        Returns:
            预测的类别
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数据

        Returns:
            预测的概率
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if not self._is_fitted:
            return {}

        return {
            "coef": self._model.coef_.tolist(),
            "intercept": self._model.intercept_.tolist(),
            "classes": self._classes.tolist(),
            "feature_names": self._feature_names,
        }

    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'LogisticRegressionModel':
        """加载模型"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {path}")
        return model


class LogisticRegressionTrainer:
    """
    逻辑回归训练器

    提供模型训练和评估功能。
    """

    def __init__(self, model_params: Optional[Dict] = None,
                 train_params: Optional[Dict] = None):
        """
        初始化训练器

        Args:
            model_params: 模型参数
            train_params: 训练参数
        """
        self.model_params = model_params or {}
        self.train_params = train_params or {}
        self._model = None
        self._metrics = {}

    def train(self, X_train: Union[np.ndarray, pd.DataFrame],
              y_train: Union[np.ndarray, pd.Series],
              X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              y_val: Optional[Union[np.ndarray, pd.Series]] = None) -> LogisticRegressionModel:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练好的模型
        """
        logger.info("Starting model training")

        # 创建模型
        self._model = LogisticRegressionModel(**self.model_params)

        # 训练模型
        self._model.fit(X_train, y_train)

        # 评估训练集
        train_metrics = self._evaluate(X_train, y_train, prefix='train')
        self._metrics.update(train_metrics)

        # 评估验证集
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, prefix='val')
            self._metrics.update(val_metrics)

        logger.info(f"Training completed. Metrics: {self._metrics}")
        return self._model

    def _evaluate(self, X: Union[np.ndarray, pd.DataFrame],
                  y: Union[np.ndarray, pd.Series],
                  prefix: str = '') -> Dict[str, float]:
        """评估模型"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        except ImportError:
            return {}

        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self._model.predict(X)
        y_proba = self._model.predict_proba(X)

        metrics = {
            f'{prefix}_accuracy': accuracy_score(y, y_pred),
        }

        # 二分类指标
        if len(self._model._classes) == 2:
            metrics[f'{prefix}_precision'] = precision_score(y, y_pred, average='binary')
            metrics[f'{prefix}_recall'] = recall_score(y, y_pred, average='binary')
            metrics[f'{prefix}_f1'] = f1_score(y, y_pred, average='binary')
            try:
                metrics[f'{prefix}_auc'] = roc_auc_score(y, y_proba[:, 1])
            except:
                pass
        else:
            metrics[f'{prefix}_precision'] = precision_score(y, y_pred, average='weighted')
            metrics[f'{prefix}_recall'] = recall_score(y, y_pred, average='weighted')
            metrics[f'{prefix}_f1'] = f1_score(y, y_pred, average='weighted')

        return metrics

    def get_metrics(self) -> Dict[str, float]:
        """获取评估指标"""
        return self._metrics


class LogisticRegressionPredictor:
    """
    逻辑回归预测器

    使用训练好的模型进行预测。
    """

    def __init__(self, model: Optional[LogisticRegressionModel] = None,
                 model_path: Optional[str] = None):
        """
        初始化预测器

        Args:
            model: 训练好的模型
            model_path: 模型文件路径
        """
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = LogisticRegressionModel.load(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        return self._model.predict_proba(X)

    def predict_with_threshold(self, X: Union[np.ndarray, pd.DataFrame],
                                threshold: float = 0.5) -> np.ndarray:
        """
        使用自定义阈值预测（仅二分类）

        Args:
            X: 特征数据
            threshold: 分类阈值

        Returns:
            预测的类别
        """
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return self.predict(X)
