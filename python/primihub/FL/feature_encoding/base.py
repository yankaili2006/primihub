"""
Federated Learning Feature Encoding Base Classes
联邦学习特征编码基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)


class FLFeatureEncoderBase(ABC):
    """联邦学习特征编码基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        """
        Args:
            FL_type: 联邦学习类型
            role: 角色
            channel: 通信通道
        """
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合编码器"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆转换"""
        pass


class FLOneHotEncoder(FLFeatureEncoderBase):
    """
    联邦学习独热编码器

    支持跨方类别值聚合
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        handle_unknown: str = "ignore",
        sparse: bool = False,
    ):
        super().__init__(FL_type, role, channel)
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.categories_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        拟合编码器

        Args:
            X: 输入数据
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        self.categories_ = []

        for i in range(n_features):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)

        # 联邦场景下同步类别
        if self.FL_type == "H" and self.channel:
            self._sync_categories()

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        执行独热编码

        Args:
            X: 输入数据

        Returns:
            编码后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_features = []

        for i, categories in enumerate(self.categories_):
            col = X[:, i]
            encoded = np.zeros((len(col), len(categories)))

            for j, cat in enumerate(categories):
                encoded[:, j] = (col == cat).astype(float)

            encoded_features.append(encoded)

        return np.hstack(encoded_features)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        result = []
        idx = 0

        for categories in self.categories_:
            n_cats = len(categories)
            encoded_block = X[:, idx:idx + n_cats]
            decoded = categories[np.argmax(encoded_block, axis=1)]
            result.append(decoded.reshape(-1, 1))
            idx += n_cats

        return np.hstack(result)

    def _sync_categories(self):
        """同步类别"""
        if self.role == "client":
            self.channel.send("local_categories", [c.tolist() for c in self.categories_])
            global_cats = self.channel.recv("global_categories")
            self.categories_ = [np.array(c) for c in global_cats]

        elif self.role == "server":
            all_cats = self.channel.recv_all("local_categories")
            global_categories = []

            for i in range(len(self.categories_)):
                merged = set(self.categories_[i].tolist())
                for client_cats in all_cats.values():
                    if i < len(client_cats):
                        merged.update(client_cats[i])
                global_categories.append(sorted(list(merged)))

            self.categories_ = [np.array(c) for c in global_categories]
            self.channel.send_all("global_categories", global_categories)


class FLLabelEncoder(FLFeatureEncoderBase):
    """
    联邦学习标签编码器

    将类别值编码为整数
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
    ):
        super().__init__(FL_type, role, channel)
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合编码器"""
        self.classes_ = np.unique(X)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}

        # 联邦场景下同步
        if self.FL_type == "H" and self.channel:
            self._sync_classes()

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """编码"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        encoded = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            if x in self.class_to_index_:
                encoded[i] = self.class_to_index_[x]
            else:
                encoded[i] = -1  # 未知类别

        return encoded

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆转换"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        return self.classes_[X]

    def _sync_classes(self):
        """同步类别"""
        if self.role == "client":
            self.channel.send("local_classes", self.classes_.tolist())
            global_classes = self.channel.recv("global_classes")
            self.classes_ = np.array(global_classes)
            self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}

        elif self.role == "server":
            all_classes = self.channel.recv_all("local_classes")
            merged = set(self.classes_.tolist())
            for client_classes in all_classes.values():
                merged.update(client_classes)

            self.classes_ = np.array(sorted(list(merged)))
            self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
            self.channel.send_all("global_classes", self.classes_.tolist())


class FLTargetEncoder(FLFeatureEncoderBase):
    """
    联邦学习目标编码器

    使用目标变量的统计量对类别特征编码
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        smoothing: float = 1.0,
    ):
        super().__init__(FL_type, role, channel)
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_map_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合目标编码器

        Args:
            X: 类别特征
            y: 目标变量
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.global_mean_ = np.mean(y)
        self.encoding_map_ = []

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            categories = np.unique(col)
            encoding = {}

            for cat in categories:
                mask = col == cat
                n_cat = np.sum(mask)
                cat_mean = np.mean(y[mask])

                # 贝叶斯平滑
                smoothed = (n_cat * cat_mean + self.smoothing * self.global_mean_) / (
                    n_cat + self.smoothing
                )
                encoding[cat] = smoothed

            self.encoding_map_.append(encoding)

        # 联邦场景下聚合
        if self.FL_type == "H" and self.channel:
            self._sync_encoding()

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """编码"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros_like(X, dtype=np.float64)

        for col_idx, encoding in enumerate(self.encoding_map_):
            for i, val in enumerate(X[:, col_idx]):
                result[i, col_idx] = encoding.get(val, self.global_mean_)

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """目标编码不支持逆转换"""
        raise NotImplementedError("目标编码不支持逆转换")

    def _sync_encoding(self):
        """同步编码"""
        if self.role == "client":
            # 发送本地统计量
            local_stats = []
            for encoding in self.encoding_map_:
                local_stats.append({str(k): v for k, v in encoding.items()})

            self.channel.send("local_encoding", local_stats)
            self.channel.send("local_count", len(self.encoding_map_[0]) if self.encoding_map_ else 0)

            global_encoding = self.channel.recv("global_encoding")
            self.encoding_map_ = [{k: v for k, v in enc.items()} for enc in global_encoding]

        elif self.role == "server":
            all_encodings = self.channel.recv_all("local_encoding")
            all_counts = self.channel.recv_all("local_count")

            # 聚合编码（加权平均）
            # 简化实现：直接使用本地编码
            global_encoding = [{str(k): v for k, v in enc.items()} for enc in self.encoding_map_]
            self.channel.send_all("global_encoding", global_encoding)


class FLHashEncoder(FLFeatureEncoderBase):
    """
    联邦学习哈希编码器

    使用哈希函数对高基数类别特征编码
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_components: int = 8,
        hash_method: str = "md5",
    ):
        super().__init__(FL_type, role, channel)
        self.n_components = n_components
        self.hash_method = hash_method

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """哈希编码不需要拟合"""
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        执行哈希编码

        Args:
            X: 输入数据

        Returns:
            哈希编码后的数据
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.zeros((X.shape[0], X.shape[1] * self.n_components))

        for col_idx in range(X.shape[1]):
            for row_idx, val in enumerate(X[:, col_idx]):
                hash_val = self._hash(str(val))
                start_idx = col_idx * self.n_components
                result[row_idx, start_idx:start_idx + self.n_components] = hash_val

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """哈希编码不支持逆转换"""
        raise NotImplementedError("哈希编码不支持逆转换")

    def _hash(self, value: str) -> np.ndarray:
        """计算哈希值"""
        if self.hash_method == "md5":
            h = hashlib.md5(value.encode()).hexdigest()
        else:
            h = hashlib.sha256(value.encode()).hexdigest()

        # 将哈希值转换为数值数组
        result = np.zeros(self.n_components)
        for i in range(self.n_components):
            idx = (i * 2) % len(h)
            result[i] = int(h[idx:idx + 2], 16) / 255.0

        return result


class FLEmbeddingEncoder(FLFeatureEncoderBase):
    """
    联邦学习嵌入编码器

    将类别特征编码为稠密向量
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        embedding_dim: int = 8,
        init_method: str = "random",
    ):
        super().__init__(FL_type, role, channel)
        self.embedding_dim = embedding_dim
        self.init_method = init_method
        self.embeddings_ = None
        self.categories_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        初始化嵌入矩阵

        Args:
            X: 类别特征
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        self.embeddings_ = []

        for col_idx in range(X.shape[1]):
            categories = np.unique(X[:, col_idx])
            self.categories_.append(categories)

            n_categories = len(categories)
            if self.init_method == "random":
                embedding = np.random.randn(n_categories, self.embedding_dim) * 0.01
            else:
                embedding = np.zeros((n_categories, self.embedding_dim))

            self.embeddings_.append(embedding)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        嵌入编码

        Args:
            X: 输入数据

        Returns:
            嵌入后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = []

        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            embedding = self.embeddings_[col_idx]
            cat_to_idx = {c: i for i, c in enumerate(categories)}

            col_embedding = np.zeros((X.shape[0], self.embedding_dim))
            for row_idx, val in enumerate(X[:, col_idx]):
                if val in cat_to_idx:
                    col_embedding[row_idx] = embedding[cat_to_idx[val]]

            result.append(col_embedding)

        return np.hstack(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """嵌入编码不支持精确逆转换"""
        raise NotImplementedError("嵌入编码不支持精确逆转换")
