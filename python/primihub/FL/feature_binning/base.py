"""
Federated Learning Feature Binning Base Classes
联邦学习特征装仓基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureBinningBase(ABC):
    """特征分箱基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_bins: int = 10,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self.n_bins = n_bins
        self.bin_edges_ = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合分箱器"""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用分箱"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.int64)

        for col in range(X.shape[1]):
            if col < len(self.bin_edges_):
                result[:, col] = np.digitize(X[:, col], self.bin_edges_[col]) - 1
                result[:, col] = np.clip(result[:, col], 0, len(self.bin_edges_[col]) - 2)

        return result

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class EqualWidthBinning(FeatureBinningBase):
    """
    等宽分箱

    将特征值范围等分为若干个区间
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算等宽分箱边界"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.bin_edges_ = []

        for col in range(X.shape[1]):
            col_data = X[:, col]
            col_min, col_max = np.nanmin(col_data), np.nanmax(col_data)
            edges = np.linspace(col_min, col_max, self.n_bins + 1)
            self.bin_edges_.append(edges)

        # 联邦场景下同步
        if self.FL_type == "H" and self.channel:
            self._sync_bin_edges()

        self._is_fitted = True
        return self

    def _sync_bin_edges(self):
        """同步分箱边界"""
        if self.role == "client":
            local_mins = [e[0] for e in self.bin_edges_]
            local_maxs = [e[-1] for e in self.bin_edges_]

            self.channel.send("local_mins", local_mins)
            self.channel.send("local_maxs", local_maxs)

            global_edges = self.channel.recv("global_bin_edges")
            self.bin_edges_ = [np.array(e) for e in global_edges]

        elif self.role == "server":
            all_mins = self.channel.recv_all("local_mins")
            all_maxs = self.channel.recv_all("local_maxs")

            global_edges = []
            for col in range(len(self.bin_edges_)):
                col_mins = [self.bin_edges_[col][0]]
                col_maxs = [self.bin_edges_[col][-1]]

                for party in all_mins:
                    if col < len(all_mins[party]):
                        col_mins.append(all_mins[party][col])
                        col_maxs.append(all_maxs[party][col])

                global_min = min(col_mins)
                global_max = max(col_maxs)
                edges = np.linspace(global_min, global_max, self.n_bins + 1)
                global_edges.append(edges)

            self.bin_edges_ = global_edges
            self.channel.send_all("global_bin_edges", [e.tolist() for e in global_edges])


class EqualFrequencyBinning(FeatureBinningBase):
    """
    等频分箱

    确保每个箱中的样本数量大致相等
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """计算等频分箱边界"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.bin_edges_ = []
        quantiles = np.linspace(0, 100, self.n_bins + 1)

        for col in range(X.shape[1]):
            col_data = X[:, col]
            edges = np.percentile(col_data[~np.isnan(col_data)], quantiles)
            edges = np.unique(edges)  # 移除重复边界
            self.bin_edges_.append(edges)

        self._is_fitted = True
        return self


class OptimalBinning(FeatureBinningBase):
    """
    最优分箱

    基于目标变量优化分箱边界
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_bins: int = 10,
        min_samples_bin: int = 50,
    ):
        super().__init__(FL_type, role, channel, n_bins)
        self.min_samples_bin = min_samples_bin

    def fit(self, X: np.ndarray, y: np.ndarray):
        """计算最优分箱边界"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.bin_edges_ = []

        for col in range(X.shape[1]):
            edges = self._find_optimal_bins(X[:, col], y)
            self.bin_edges_.append(edges)

        self._is_fitted = True
        return self

    def _find_optimal_bins(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用贪婪算法找最优分箱"""
        # 初始化：等频分箱
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(x[~np.isnan(x)], quantiles)
        edges = np.unique(edges)

        # 简化：使用卡方检验合并相邻箱
        while len(edges) > self.n_bins + 1:
            chi_scores = []
            for i in range(len(edges) - 2):
                mask1 = (x >= edges[i]) & (x < edges[i + 1])
                mask2 = (x >= edges[i + 1]) & (x < edges[i + 2])

                if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                    p1 = np.mean(y[mask1])
                    p2 = np.mean(y[mask2])
                    n1, n2 = np.sum(mask1), np.sum(mask2)

                    # 卡方统计量
                    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
                    chi = ((p1 - p2) ** 2) / (p_pool * (1 - p_pool + 1e-10) / n1 + p_pool * (1 - p_pool + 1e-10) / n2 + 1e-10)
                    chi_scores.append(chi)
                else:
                    chi_scores.append(float('inf'))

            # 合并卡方值最小的相邻箱
            min_idx = np.argmin(chi_scores)
            edges = np.delete(edges, min_idx + 1)

        return edges


class WOEBinning(FeatureBinningBase):
    """
    WOE分箱

    使用Weight of Evidence进行分箱
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_bins: int = 10,
    ):
        super().__init__(FL_type, role, channel, n_bins)
        self.woe_values_ = None
        self.iv_values_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """计算WOE分箱"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.bin_edges_ = []
        self.woe_values_ = []
        self.iv_values_ = []

        # 先做等频分箱
        quantiles = np.linspace(0, 100, self.n_bins + 1)

        for col in range(X.shape[1]):
            col_data = X[:, col]
            edges = np.percentile(col_data[~np.isnan(col_data)], quantiles)
            edges = np.unique(edges)
            self.bin_edges_.append(edges)

            # 计算WOE和IV
            woe, iv = self._compute_woe_iv(col_data, y, edges)
            self.woe_values_.append(woe)
            self.iv_values_.append(iv)

        self._is_fitted = True
        return self

    def _compute_woe_iv(
        self, x: np.ndarray, y: np.ndarray, edges: np.ndarray
    ) -> Tuple[Dict[int, float], float]:
        """计算WOE和IV值"""
        total_good = np.sum(y == 0)
        total_bad = np.sum(y == 1)

        woe = {}
        iv = 0.0

        bins = np.digitize(x, edges) - 1

        for b in range(len(edges) - 1):
            mask = bins == b
            good = np.sum((y == 0) & mask)
            bad = np.sum((y == 1) & mask)

            good_pct = (good + 0.5) / (total_good + 1)
            bad_pct = (bad + 0.5) / (total_bad + 1)

            woe[b] = np.log(good_pct / bad_pct)
            iv += (good_pct - bad_pct) * woe[b]

        return woe, iv

    def transform_woe(self, X: np.ndarray) -> np.ndarray:
        """将特征值转换为WOE值"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col in range(X.shape[1]):
            if col < len(self.bin_edges_):
                bins = np.digitize(X[:, col], self.bin_edges_[col]) - 1
                bins = np.clip(bins, 0, len(self.bin_edges_[col]) - 2)

                for i, b in enumerate(bins):
                    result[i, col] = self.woe_values_[col].get(b, 0)

        return result
