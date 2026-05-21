"""
Feature Binning Base Classes
特征分箱基础类

提供各种特征分箱功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBinnerBase(ABC):
    """特征分箱基类"""

    def __init__(self, columns: Optional[List[str]] = None, n_bins: int = 5):
        """
        初始化特征分箱器

        Args:
            columns: 要分箱的列，None表示所有数值列
            n_bins: 分箱数量
        """
        self.columns = columns
        self.n_bins = n_bins
        self._is_fitted = False
        self._bin_edges = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureBinnerBase':
        """拟合分箱器"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        pass

    def fit_transform(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(data, y)
        return self.transform(data)

    def get_bin_edges(self) -> Dict[str, List[float]]:
        """获取分箱边界"""
        return self._bin_edges

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """获取要分箱的数值列"""
        if self.columns:
            return [c for c in self.columns if c in data.columns]
        return data.select_dtypes(include=[np.number]).columns.tolist()


class EqualWidthBinner(FeatureBinnerBase):
    """
    等宽分箱器

    将数据按等间隔划分为指定数量的箱。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 5,
                 labels: Optional[List] = None):
        """
        初始化等宽分箱器

        Args:
            columns: 要分箱的列
            n_bins: 分箱数量
            labels: 分箱标签
        """
        super().__init__(columns, n_bins)
        self.labels = labels

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EqualWidthBinner':
        """拟合等宽分箱器"""
        logger.info(f"Fitting EqualWidthBinner with n_bins={self.n_bins}")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            min_val, max_val = col_data.min(), col_data.max()
            bin_width = (max_val - min_val) / self.n_bins

            edges = [min_val + i * bin_width for i in range(self.n_bins + 1)]
            edges[0] = -np.inf  # 包含最小值
            edges[-1] = np.inf  # 包含最大值

            self._bin_edges[col] = edges

        self._is_fitted = True
        logger.info(f"Fitted EqualWidthBinner for {len(cols)} columns")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """等宽分箱转换"""
        if not self._is_fitted:
            raise RuntimeError("EqualWidthBinner is not fitted")

        result = data.copy()
        labels = self.labels if self.labels else list(range(self.n_bins))

        for col, edges in self._bin_edges.items():
            if col not in result.columns:
                continue

            result[col] = pd.cut(result[col], bins=edges, labels=labels, include_lowest=True)

        return result


class EqualFrequencyBinner(FeatureBinnerBase):
    """
    等频分箱器

    将数据按分位数划分为指定数量的箱，每个箱包含相近数量的样本。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 5,
                 labels: Optional[List] = None,
                 duplicates: str = 'drop'):
        """
        初始化等频分箱器

        Args:
            columns: 要分箱的列
            n_bins: 分箱数量
            labels: 分箱标签
            duplicates: 重复边界处理方式
        """
        super().__init__(columns, n_bins)
        self.labels = labels
        self.duplicates = duplicates

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EqualFrequencyBinner':
        """拟合等频分箱器"""
        logger.info(f"Fitting EqualFrequencyBinner with n_bins={self.n_bins}")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            try:
                _, edges = pd.qcut(col_data, q=self.n_bins, retbins=True, duplicates=self.duplicates)
                edges = edges.tolist()
                edges[0] = -np.inf
                edges[-1] = np.inf
                self._bin_edges[col] = edges
            except ValueError as e:
                logger.warning(f"Could not fit bins for {col}: {e}")
                # 使用等宽作为备选
                min_val, max_val = col_data.min(), col_data.max()
                edges = np.linspace(min_val, max_val, self.n_bins + 1).tolist()
                edges[0] = -np.inf
                edges[-1] = np.inf
                self._bin_edges[col] = edges

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """等频分箱转换"""
        if not self._is_fitted:
            raise RuntimeError("EqualFrequencyBinner is not fitted")

        result = data.copy()

        for col, edges in self._bin_edges.items():
            if col not in result.columns:
                continue

            n_labels = len(edges) - 1
            labels = self.labels[:n_labels] if self.labels else list(range(n_labels))
            result[col] = pd.cut(result[col], bins=edges, labels=labels, include_lowest=True)

        return result


class KMeansBinner(FeatureBinnerBase):
    """
    K-Means分箱器

    使用K-Means聚类进行分箱。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 5,
                 random_state: int = 42):
        """
        初始化K-Means分箱器

        Args:
            columns: 要分箱的列
            n_bins: 分箱数量（聚类数）
            random_state: 随机种子
        """
        super().__init__(columns, n_bins)
        self.random_state = random_state
        self._cluster_centers = {}

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'KMeansBinner':
        """拟合K-Means分箱器"""
        logger.info(f"Fitting KMeansBinner with n_bins={self.n_bins}")

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("sklearn is required for KMeansBinner")
            raise ImportError("sklearn is required for KMeansBinner")

        cols = self._get_numeric_columns(data)

        for col in cols:
            col_data = data[col].dropna().values.reshape(-1, 1)
            if len(col_data) < self.n_bins:
                continue

            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init=10)
            kmeans.fit(col_data)

            # 排序聚类中心
            centers = sorted(kmeans.cluster_centers_.flatten())
            self._cluster_centers[col] = centers

            # 计算边界（相邻中心的中点）
            edges = [-np.inf]
            for i in range(len(centers) - 1):
                edges.append((centers[i] + centers[i + 1]) / 2)
            edges.append(np.inf)

            self._bin_edges[col] = edges

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """K-Means分箱转换"""
        if not self._is_fitted:
            raise RuntimeError("KMeansBinner is not fitted")

        result = data.copy()

        for col, edges in self._bin_edges.items():
            if col not in result.columns:
                continue

            result[col] = pd.cut(result[col], bins=edges, labels=range(len(edges) - 1))

        return result


class DecisionTreeBinner(FeatureBinnerBase):
    """
    决策树分箱器

    使用决策树进行有监督分箱。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 n_bins: int = 5,
                 min_samples_leaf: int = 50,
                 random_state: int = 42):
        """
        初始化决策树分箱器

        Args:
            columns: 要分箱的列
            n_bins: 最大分箱数（树的最大叶子数）
            min_samples_leaf: 叶子节点最小样本数
            random_state: 随机种子
        """
        super().__init__(columns, n_bins)
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DecisionTreeBinner':
        """拟合决策树分箱器"""
        if y is None:
            raise ValueError("DecisionTreeBinner requires target variable y")

        logger.info(f"Fitting DecisionTreeBinner with max_leaf_nodes={self.n_bins}")

        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except ImportError:
            raise ImportError("sklearn is required for DecisionTreeBinner")

        cols = self._get_numeric_columns(data)

        # 根据目标变量类型选择分类或回归树
        if y.dtype in ['int64', 'object', 'category'] or y.nunique() <= 10:
            TreeModel = DecisionTreeClassifier
        else:
            TreeModel = DecisionTreeRegressor

        for col in cols:
            col_data = data[[col]].dropna()
            y_aligned = y.loc[col_data.index]

            if len(col_data) < self.min_samples_leaf * 2:
                continue

            tree = TreeModel(
                max_leaf_nodes=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(col_data, y_aligned)

            # 提取分割点
            thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
            thresholds = sorted([t for t in thresholds if t != -2.0])

            edges = [-np.inf] + thresholds + [np.inf]
            self._bin_edges[col] = edges

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """决策树分箱转换"""
        if not self._is_fitted:
            raise RuntimeError("DecisionTreeBinner is not fitted")

        result = data.copy()

        for col, edges in self._bin_edges.items():
            if col not in result.columns:
                continue

            result[col] = pd.cut(result[col], bins=edges, labels=range(len(edges) - 1))

        return result


class CustomBinner(FeatureBinnerBase):
    """
    自定义分箱器

    使用用户指定的边界进行分箱。
    """

    def __init__(self, bin_edges: Dict[str, List[float]],
                 labels: Optional[Dict[str, List]] = None):
        """
        初始化自定义分箱器

        Args:
            bin_edges: 每列的分箱边界，格式为 {列名: [边界1, 边界2, ...]}
            labels: 每列的分箱标签
        """
        super().__init__(columns=list(bin_edges.keys()))
        self._bin_edges = bin_edges
        self.custom_labels = labels or {}
        self._is_fitted = True

    def fit(self, data: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CustomBinner':
        """自定义分箱器不需要拟合"""
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """自定义分箱转换"""
        result = data.copy()

        for col, edges in self._bin_edges.items():
            if col not in result.columns:
                continue

            n_bins = len(edges) - 1
            labels = self.custom_labels.get(col, list(range(n_bins)))
            result[col] = pd.cut(result[col], bins=edges, labels=labels, include_lowest=True)

        return result
