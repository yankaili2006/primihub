"""
XGBoost Base Classes
XGBoost基础类

提供XGBoost模型的训练和预测功能。
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost模型

    支持分类和回归任务。
    """

    def __init__(self,
                 task: str = 'classification',
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 min_child_weight: int = 1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 objective: Optional[str] = None,
                 eval_metric: Optional[str] = None,
                 scale_pos_weight: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: int = 0):
        """
        初始化XGBoost模型

        Args:
            task: 任务类型（classification, regression）
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
            min_child_weight: 最小子节点权重
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            reg_alpha: L1正则化
            reg_lambda: L2正则化
            objective: 目标函数
            eval_metric: 评估指标
            scale_pos_weight: 正样本权重
            random_state: 随机种子
            n_jobs: 并行数
            verbose: 日志详细程度
        """
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self._model = None
        self._is_fitted = False
        self._feature_names = None
        self._feature_importances = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            eval_set: Optional[List[Tuple]] = None,
            early_stopping_rounds: Optional[int] = None) -> 'XGBoostModel':
        """
        训练XGBoost模型

        Args:
            X: 特征数据
            y: 标签数据
            eval_set: 验证集列表
            early_stopping_rounds: 早停轮数

        Returns:
            self
        """
        logger.info(f"Training XGBoost model for {self.task}")

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostModel")

        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 确定目标函数
        if self.objective is None:
            if self.task == 'classification':
                n_classes = len(np.unique(y))
                if n_classes == 2:
                    objective = 'binary:logistic'
                else:
                    objective = 'multi:softprob'
            else:
                objective = 'reg:squarederror'
        else:
            objective = self.objective

        # 创建模型
        if self.task == 'classification':
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective=objective,
                eval_metric=self.eval_metric,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=self.verbose,
            )
        else:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective=objective,
                eval_metric=self.eval_metric,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=self.verbose,
            )

        # 训练模型
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self._model.fit(X, y, **fit_params)

        # 获取特征重要性
        if self._feature_names:
            self._feature_importances = dict(zip(
                self._feature_names,
                self._model.feature_importances_
            ))

        self._is_fitted = True
        logger.info("Model trained successfully")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率（仅分类任务）

        Args:
            X: 特征数据

        Returns:
            预测的概率
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        if self.task != 'classification':
            raise RuntimeError("predict_proba is only available for classification")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self._feature_importances

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "task": self.task,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_names": self._feature_names,
            "feature_importances": self._feature_importances,
        }

    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """加载模型"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {path}")
        return model


class XGBoostTrainer:
    """
    XGBoost训练器

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
              y_val: Optional[Union[np.ndarray, pd.Series]] = None) -> XGBoostModel:
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
        logger.info("Starting XGBoost training")

        # 创建模型
        self._model = XGBoostModel(**self.model_params)

        # 准备验证集
        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_np = X_val.values
            else:
                X_val_np = X_val
            if isinstance(y_val, pd.Series):
                y_val_np = y_val.values
            else:
                y_val_np = y_val
            eval_set = [(X_val_np, y_val_np)]

        # 训练模型
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.train_params.get('early_stopping_rounds')
        )

        # 评估
        task = self.model_params.get('task', 'classification')
        train_metrics = self._evaluate(X_train, y_train, task, prefix='train')
        self._metrics.update(train_metrics)

        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, task, prefix='val')
            self._metrics.update(val_metrics)

        logger.info(f"Training completed. Metrics: {self._metrics}")
        return self._model

    def _evaluate(self, X: Union[np.ndarray, pd.DataFrame],
                  y: Union[np.ndarray, pd.Series],
                  task: str,
                  prefix: str = '') -> Dict[str, float]:
        """评估模型"""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
            )
        except ImportError:
            return {}

        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self._model.predict(X)

        metrics = {}

        if task == 'classification':
            metrics[f'{prefix}_accuracy'] = accuracy_score(y, y_pred)

            n_classes = len(np.unique(y))
            if n_classes == 2:
                metrics[f'{prefix}_precision'] = precision_score(y, y_pred)
                metrics[f'{prefix}_recall'] = recall_score(y, y_pred)
                metrics[f'{prefix}_f1'] = f1_score(y, y_pred)
                try:
                    y_proba = self._model.predict_proba(X)
                    metrics[f'{prefix}_auc'] = roc_auc_score(y, y_proba[:, 1])
                except:
                    pass
            else:
                metrics[f'{prefix}_precision'] = precision_score(y, y_pred, average='weighted')
                metrics[f'{prefix}_recall'] = recall_score(y, y_pred, average='weighted')
                metrics[f'{prefix}_f1'] = f1_score(y, y_pred, average='weighted')
        else:
            metrics[f'{prefix}_mse'] = mean_squared_error(y, y_pred)
            metrics[f'{prefix}_rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics[f'{prefix}_mae'] = mean_absolute_error(y, y_pred)
            metrics[f'{prefix}_r2'] = r2_score(y, y_pred)

        return metrics

    def get_metrics(self) -> Dict[str, float]:
        """获取评估指标"""
        return self._metrics


class XGBoostPredictor:
    """
    XGBoost预测器

    使用训练好的模型进行预测。
    """

    def __init__(self, model: Optional[XGBoostModel] = None,
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
            self._model = XGBoostModel.load(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测"""
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率（仅分类）"""
        return self._model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self._model.get_feature_importances()
