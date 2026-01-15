"""
Federated Learning Data Splitting Base Classes
联邦学习数据分割基础类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


class DataSplittingBase(ABC):
    """数据分割基类"""

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        random_state: Optional[int] = None,
    ):
        self.FL_type = FL_type
        self.role = role
        self.channel = channel
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """分割数据"""
        pass


class TrainTestSplitter(DataSplittingBase):
    """
    训练测试分割

    将数据分割为训练集和测试集
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(FL_type, role, channel, random_state)
        self.test_size = test_size
        self.shuffle = shuffle

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        分割数据

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # 联邦场景下同步随机种子
        if self.FL_type == "H" and self.channel:
            indices = self._sync_shuffle_indices(indices)
        elif self.shuffle:
            np.random.shuffle(indices)

        n_test = int(n_samples * self.test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, X_test = X[train_indices], X[test_indices]

        if y is not None:
            y_train, y_test = y[train_indices], y[test_indices]
        else:
            y_train, y_test = None, None

        return (X_train, y_train), (X_test, y_test)

    def _sync_shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """同步随机打乱索引"""
        if self.role == "server":
            if self.shuffle:
                np.random.shuffle(indices)
            self.channel.send_all("shuffle_seed", self.random_state or int(np.random.randint(0, 10000)))
        elif self.role == "client":
            seed = self.channel.recv("shuffle_seed")
            np.random.seed(seed)
            if self.shuffle:
                np.random.shuffle(indices)

        return indices


class KFoldSplitter(DataSplittingBase):
    """
    K折交叉验证分割

    将数据分割为K个折叠
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(FL_type, role, channel, random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成K折分割

        Yields:
            (train_indices, test_indices) for each fold
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            yield train_indices, test_indices
            current = stop

    def get_n_splits(self) -> int:
        """返回折数"""
        return self.n_splits


class StratifiedSplitter(DataSplittingBase):
    """
    分层分割

    保持标签分布的分割
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(FL_type, role, channel, random_state)
        self.test_size = test_size
        self.shuffle = shuffle

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        分层分割数据

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        classes, y_indices = np.unique(y, return_inverse=True)

        train_indices = []
        test_indices = []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]

            if self.shuffle:
                np.random.shuffle(cls_indices)

            n_test = int(len(cls_indices) * self.test_size)

            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if self.shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return (X_train, y_train), (X_test, y_test)


class TimeSplitter(DataSplittingBase):
    """
    时间序列分割

    按时间顺序分割数据
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Args:
            n_splits: 分割数
            test_size: 测试集大小
            gap: 训练集和测试集之间的间隔
        """
        super().__init__(FL_type, role, channel)
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        时间序列分割

        Yields:
            (train_indices, test_indices) for each split
        """
        n_samples = X.shape[0]
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size

            train_end = test_start - self.gap

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


class GroupSplitter(DataSplittingBase):
    """
    分组分割

    按组分割数据，确保同一组的数据在同一集合中
    """

    def __init__(
        self,
        FL_type: str = "H",
        role: Optional[str] = None,
        channel: Optional[Any] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
    ):
        super().__init__(FL_type, role, channel, random_state)
        self.test_size = test_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: np.ndarray = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        按组分割数据

        Args:
            X: 特征
            y: 标签
            groups: 组标签

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        if groups is None:
            # 如果没有组信息，退化为普通分割
            splitter = TrainTestSplitter(
                test_size=self.test_size, random_state=self.random_state
            )
            return splitter.split(X, y)

        unique_groups = np.unique(groups)
        np.random.shuffle(unique_groups)

        n_test_groups = int(len(unique_groups) * self.test_size)
        test_groups = set(unique_groups[:n_test_groups])

        train_mask = np.array([g not in test_groups for g in groups])
        test_mask = ~train_mask

        X_train, X_test = X[train_mask], X[test_mask]

        if y is not None:
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            y_train, y_test = None, None

        return (X_train, y_train), (X_test, y_test)
