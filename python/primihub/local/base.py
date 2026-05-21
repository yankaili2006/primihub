"""
Local Processing Base Classes
单方本地处理基类

提供所有单方本地处理模块的基础类定义。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LocalBaseModel(ABC):
    """
    单方本地处理模型基类

    所有单方处理模块都需要继承此类并实现run()方法。

    Attributes:
        common_params: 通用参数
        role_params: 角色特定参数
        node_info: 节点信息
        task_info: 任务信息
    """

    def __init__(self, **kwargs):
        """
        初始化单方处理模型

        Args:
            **kwargs: 包含以下键的字典:
                - common_params: 通用参数
                - role_params: 角色特定参数
                - node_info: 节点信息
                - task_info: 任务信息
        """
        self.common_params = kwargs.get('common_params', {})
        self.role_params = kwargs.get('role_params', {})
        self.node_info = kwargs.get('node_info', {})
        self.task_info = kwargs.get('task_info', {})

        # 解析参数
        self._parse_params()

    @abstractmethod
    def _parse_params(self):
        """解析参数，子类必须实现"""
        pass

    @abstractmethod
    def run(self) -> Any:
        """执行处理，子类必须实现"""
        pass

    def _load_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        加载数据的通用方法

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: 数据和标签
        """
        from primihub.FL.utils.dataset import read_data

        data_info = self.role_params.get('data', {})
        selected_columns = self.role_params.get('selected_columns', None)
        label_column = self.role_params.get('label_column', None)

        if not data_info:
            logger.warning("No data info provided")
            return pd.DataFrame(), None

        # 读取数据
        data = read_data(data_info=data_info, selected_column=selected_columns)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # 提取标签
        labels = None
        if label_column and label_column in data.columns:
            labels = data[label_column]
            data = data.drop(columns=[label_column])

        return data, labels

    def _save_result(self, result: Any, output_path: str = None):
        """
        保存结果

        Args:
            result: 要保存的结果
            output_path: 输出路径，如果为None则使用默认路径
        """
        from primihub.FL.utils.file import save_pickle_file, save_json_file

        if output_path is None:
            output_path = self.common_params.get('output_path', '')

        if not output_path:
            logger.warning("No output path specified, result not saved")
            return

        if output_path.endswith('.json'):
            save_json_file(result, output_path)
        else:
            save_pickle_file(result, output_path)

        logger.info(f"Result saved to: {output_path}")


class LocalProcessorBase(ABC):
    """
    单方本地处理器基类

    提供fit/transform接口的处理器基类。

    Attributes:
        _is_fitted: 是否已拟合
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'LocalProcessorBase':
        """
        拟合处理器

        Args:
            X: 输入数据
            y: 标签（可选）

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        转换数据

        Args:
            X: 输入数据

        Returns:
            转换后的数据
        """
        pass

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        拟合并转换数据

        Args:
            X: 输入数据
            y: 标签（可选）

        Returns:
            转换后的数据
        """
        self.fit(X, y)
        return self.transform(X)

    def check_is_fitted(self):
        """检查是否已拟合"""
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted. Call fit() first.")

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        验证并转换输入数据

        Args:
            X: 输入数据

        Returns:
            pd.DataFrame格式的数据
        """
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(X)}")
