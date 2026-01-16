"""
Feature Selection Base Classes
特征筛选基础类

提供各种特征筛选功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelectorBase(ABC):
    """特征筛选基类"""

    def __init__(self, n_features: Optional[int] = None,
                 threshold: Optional[float] = None):
        """
        初始化特征筛选器

        Args:
            n_features: 要选择的特征数量
            threshold: 筛选阈值
        """
        self.n_features = n_features
        self.threshold = threshold
        self._is_fitted = False
        self._selected_features = []
        self._feature_scores = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelectorBase':
        """拟合筛选器"""
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """选择特征"""
        if not self._is_fitted:
            raise RuntimeError("Feature selector is not fitted")

        return data[self._selected_features]

    def fit_transform(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并选择特征"""
        self.fit(data, y)
        return self.transform(data)

    def get_selected_features(self) -> List[str]:
        """获取选中的特征"""
        return self._selected_features

    def get_feature_scores(self) -> Dict[str, float]:
        """获取特征得分"""
        return self._feature_scores


class VarianceSelector(FeatureSelectorBase):
    """
    方差筛选器

    删除方差低于阈值的特征。
    """

    def __init__(self, threshold: float = 0.0):
        """
        初始化方差筛选器

        Args:
            threshold: 方差阈值
        """
        super().__init__(threshold=threshold)

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VarianceSelector':
        """拟合方差筛选器"""
        logger.info(f"Fitting VarianceSelector with threshold={self.threshold}")

        numeric_data = data.select_dtypes(include=[np.number])

        for col in numeric_data.columns:
            variance = numeric_data[col].var()
            self._feature_scores[col] = variance

            if variance > self.threshold:
                self._selected_features.append(col)

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + self._selected_features

        self._is_fitted = True
        logger.info(f"Selected {len(self._selected_features)} features")
        return self


class CorrelationSelector(FeatureSelectorBase):
    """
    相关性筛选器

    删除与目标变量相关性低于阈值的特征，或删除高度相关的冗余特征。
    """

    def __init__(self, threshold: float = 0.1,
                 method: str = "target",
                 correlation_method: str = "pearson"):
        """
        初始化相关性筛选器

        Args:
            threshold: 相关性阈值
            method: 筛选方法
                - target: 根据与目标变量的相关性筛选
                - redundancy: 删除特征间高度相关的冗余特征
            correlation_method: 相关性计算方法
        """
        super().__init__(threshold=threshold)
        self.method = method
        self.correlation_method = correlation_method

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationSelector':
        """拟合相关性筛选器"""
        logger.info(f"Fitting CorrelationSelector with method={self.method}")

        numeric_data = data.select_dtypes(include=[np.number])

        if self.method == "target":
            if y is None:
                raise ValueError("Target variable required for target correlation method")

            for col in numeric_data.columns:
                corr = numeric_data[col].corr(y, method=self.correlation_method)
                self._feature_scores[col] = abs(corr)

                if abs(corr) >= self.threshold:
                    self._selected_features.append(col)

        elif self.method == "redundancy":
            # 计算特征间相关性矩阵
            corr_matrix = numeric_data.corr(method=self.correlation_method)
            selected = set(numeric_data.columns)

            for i, col1 in enumerate(numeric_data.columns):
                if col1 not in selected:
                    continue

                for col2 in numeric_data.columns[i + 1:]:
                    if col2 not in selected:
                        continue

                    if abs(corr_matrix.loc[col1, col2]) > self.threshold:
                        # 删除方差较小的特征
                        if numeric_data[col1].var() >= numeric_data[col2].var():
                            selected.discard(col2)
                        else:
                            selected.discard(col1)
                            break

            self._selected_features = list(selected)

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + self._selected_features

        self._is_fitted = True
        logger.info(f"Selected {len(self._selected_features)} features")
        return self


class MutualInfoSelector(FeatureSelectorBase):
    """
    互信息筛选器

    基于与目标变量的互信息选择特征。
    """

    def __init__(self, n_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 task: str = "auto",
                 random_state: int = 42):
        """
        初始化互信息筛选器

        Args:
            n_features: 要选择的特征数量
            threshold: 互信息阈值
            task: 任务类型（classification, regression, auto）
            random_state: 随机种子
        """
        super().__init__(n_features=n_features, threshold=threshold)
        self.task = task
        self.random_state = random_state

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MutualInfoSelector':
        """拟合互信息筛选器"""
        if y is None:
            raise ValueError("MutualInfoSelector requires target variable y")

        logger.info("Fitting MutualInfoSelector")

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            raise ImportError("sklearn is required for MutualInfoSelector")

        numeric_data = data.select_dtypes(include=[np.number])

        # 确定任务类型
        task = self.task
        if task == "auto":
            if y.dtype in ['int64', 'object', 'category'] or y.nunique() <= 10:
                task = "classification"
            else:
                task = "regression"

        # 计算互信息
        if task == "classification":
            mi_scores = mutual_info_classif(numeric_data, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(numeric_data, y, random_state=self.random_state)

        for col, score in zip(numeric_data.columns, mi_scores):
            self._feature_scores[col] = score

        # 选择特征
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.n_features:
            selected = [f[0] for f in sorted_features[:self.n_features]]
        elif self.threshold:
            selected = [f[0] for f in sorted_features if f[1] >= self.threshold]
        else:
            selected = [f[0] for f in sorted_features]

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + selected

        self._is_fitted = True
        logger.info(f"Selected {len(self._selected_features)} features")
        return self


class ChiSquareSelector(FeatureSelectorBase):
    """
    卡方筛选器

    基于卡方检验选择分类特征。
    """

    def __init__(self, n_features: Optional[int] = None,
                 threshold: Optional[float] = None):
        """
        初始化卡方筛选器

        Args:
            n_features: 要选择的特征数量
            threshold: p值阈值（低于此值的特征被选中）
        """
        super().__init__(n_features=n_features, threshold=threshold)
        self._p_values = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ChiSquareSelector':
        """拟合卡方筛选器"""
        if y is None:
            raise ValueError("ChiSquareSelector requires target variable y")

        logger.info("Fitting ChiSquareSelector")

        try:
            from sklearn.feature_selection import chi2
        except ImportError:
            raise ImportError("sklearn is required for ChiSquareSelector")

        # 卡方检验需要非负值
        numeric_data = data.select_dtypes(include=[np.number])
        non_negative_data = numeric_data.clip(lower=0)

        chi2_scores, p_values = chi2(non_negative_data, y)

        for col, score, p_val in zip(numeric_data.columns, chi2_scores, p_values):
            self._feature_scores[col] = score
            self._p_values[col] = p_val

        # 选择特征
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.n_features:
            selected = [f[0] for f in sorted_features[:self.n_features]]
        elif self.threshold:
            selected = [col for col, p_val in self._p_values.items() if p_val < self.threshold]
        else:
            selected = [f[0] for f in sorted_features]

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + selected

        self._is_fitted = True
        return self


class RFESelector(FeatureSelectorBase):
    """
    递归特征消除筛选器

    使用递归特征消除选择特征。
    """

    def __init__(self, n_features: int = 10,
                 estimator: str = "logistic",
                 step: int = 1):
        """
        初始化RFE筛选器

        Args:
            n_features: 要选择的特征数量
            estimator: 基础估计器（logistic, linear, rf）
            step: 每次迭代移除的特征数
        """
        super().__init__(n_features=n_features)
        self.estimator = estimator
        self.step = step

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RFESelector':
        """拟合RFE筛选器"""
        if y is None:
            raise ValueError("RFESelector requires target variable y")

        logger.info(f"Fitting RFESelector with n_features={self.n_features}")

        try:
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("sklearn is required for RFESelector")

        numeric_data = data.select_dtypes(include=[np.number])

        # 创建估计器
        if self.estimator == "logistic":
            base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif self.estimator == "linear":
            base_estimator = LinearRegression()
        elif self.estimator == "rf":
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        # 执行RFE
        rfe = RFE(
            estimator=base_estimator,
            n_features_to_select=min(self.n_features, len(numeric_data.columns)),
            step=self.step
        )
        rfe.fit(numeric_data, y)

        # 记录特征排名
        for col, rank in zip(numeric_data.columns, rfe.ranking_):
            self._feature_scores[col] = -rank  # 排名越小越好

        # 选择特征
        selected = numeric_data.columns[rfe.support_].tolist()

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + selected

        self._is_fitted = True
        return self


class LassoSelector(FeatureSelectorBase):
    """
    Lasso筛选器

    使用Lasso回归的系数进行特征选择。
    """

    def __init__(self, alpha: float = 1.0,
                 threshold: float = 0.0,
                 n_features: Optional[int] = None):
        """
        初始化Lasso筛选器

        Args:
            alpha: Lasso正则化参数
            threshold: 系数阈值
            n_features: 要选择的特征数量
        """
        super().__init__(n_features=n_features, threshold=threshold)
        self.alpha = alpha
        self._coefficients = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LassoSelector':
        """拟合Lasso筛选器"""
        if y is None:
            raise ValueError("LassoSelector requires target variable y")

        logger.info(f"Fitting LassoSelector with alpha={self.alpha}")

        try:
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("sklearn is required for LassoSelector")

        numeric_data = data.select_dtypes(include=[np.number])

        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # 训练Lasso
        lasso = Lasso(alpha=self.alpha, random_state=42, max_iter=10000)
        lasso.fit(scaled_data, y)

        # 记录系数
        for col, coef in zip(numeric_data.columns, lasso.coef_):
            self._coefficients[col] = coef
            self._feature_scores[col] = abs(coef)

        # 选择特征
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.n_features:
            selected = [f[0] for f in sorted_features[:self.n_features]]
        else:
            selected = [f[0] for f in sorted_features if f[1] > self.threshold]

        # 保留非数值列
        non_numeric_cols = [c for c in data.columns if c not in numeric_data.columns]
        self._selected_features = non_numeric_cols + selected

        self._is_fitted = True
        return self
