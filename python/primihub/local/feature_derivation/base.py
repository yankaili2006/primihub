"""
Feature Derivation Base Classes
特征衍生基础类

提供各种特征衍生/生成功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureDeriverBase(ABC):
    """特征衍生基类"""

    def __init__(self, columns: Optional[List[str]] = None):
        """
        初始化特征衍生器

        Args:
            columns: 用于衍生的列
        """
        self.columns = columns
        self._derived_columns = []

    @abstractmethod
    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        衍生特征

        Args:
            data: 输入数据

        Returns:
            包含衍生特征的数据
        """
        pass

    def get_derived_columns(self) -> List[str]:
        """获取衍生的列名"""
        return self._derived_columns


class PolynomialDeriver(FeatureDeriverBase):
    """
    多项式特征衍生器

    生成多项式特征（如x², x³等）。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 degree: int = 2,
                 include_bias: bool = False):
        """
        初始化多项式特征衍生器

        Args:
            columns: 用于衍生的列
            degree: 多项式最高次数
            include_bias: 是否包含偏置项（常数1）
        """
        super().__init__(columns)
        self.degree = degree
        self.include_bias = include_bias

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生多项式特征"""
        logger.info(f"Deriving polynomial features with degree={self.degree}")

        result = data.copy()

        if self.columns:
            cols = [c for c in self.columns if c in data.columns]
        else:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 添加偏置项
        if self.include_bias:
            result['poly_bias'] = 1
            self._derived_columns.append('poly_bias')

        # 生成高次特征
        for col in cols:
            for d in range(2, self.degree + 1):
                new_col = f"{col}_pow{d}"
                result[new_col] = data[col] ** d
                self._derived_columns.append(new_col)

        logger.info(f"Derived {len(self._derived_columns)} polynomial features")
        return result


class InteractionDeriver(FeatureDeriverBase):
    """
    交互特征衍生器

    生成特征之间的交互项。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 interaction_type: str = "multiply",
                 max_interactions: int = 2):
        """
        初始化交互特征衍生器

        Args:
            columns: 用于交互的列
            interaction_type: 交互类型（multiply, add, subtract, divide）
            max_interactions: 最大交互特征数（组合数）
        """
        super().__init__(columns)
        self.interaction_type = interaction_type
        self.max_interactions = max_interactions

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生交互特征"""
        logger.info(f"Deriving interaction features with type={self.interaction_type}")

        result = data.copy()

        if self.columns:
            cols = [c for c in self.columns if c in data.columns]
        else:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 生成两两组合的交互特征
        for col1, col2 in combinations(cols, 2):
            if len(self._derived_columns) >= self.max_interactions * len(cols):
                break

            if self.interaction_type == "multiply":
                new_col = f"{col1}_x_{col2}"
                result[new_col] = data[col1] * data[col2]
            elif self.interaction_type == "add":
                new_col = f"{col1}_plus_{col2}"
                result[new_col] = data[col1] + data[col2]
            elif self.interaction_type == "subtract":
                new_col = f"{col1}_minus_{col2}"
                result[new_col] = data[col1] - data[col2]
            elif self.interaction_type == "divide":
                new_col = f"{col1}_div_{col2}"
                result[new_col] = data[col1] / (data[col2] + 1e-10)  # 避免除零

            self._derived_columns.append(new_col)

        logger.info(f"Derived {len(self._derived_columns)} interaction features")
        return result


class AggregationDeriver(FeatureDeriverBase):
    """
    聚合特征衍生器

    基于分组聚合生成新特征。
    """

    def __init__(self, group_columns: List[str],
                 agg_columns: List[str],
                 agg_functions: List[str] = None):
        """
        初始化聚合特征衍生器

        Args:
            group_columns: 分组列
            agg_columns: 要聚合的列
            agg_functions: 聚合函数列表（mean, sum, count, min, max, std）
        """
        super().__init__(agg_columns)
        self.group_columns = group_columns
        self.agg_columns = agg_columns
        self.agg_functions = agg_functions or ['mean', 'sum', 'count']

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生聚合特征"""
        logger.info(f"Deriving aggregation features grouped by {self.group_columns}")

        result = data.copy()

        for agg_col in self.agg_columns:
            if agg_col not in data.columns:
                continue

            for agg_func in self.agg_functions:
                new_col = f"{agg_col}_groupby_{'_'.join(self.group_columns)}_{agg_func}"

                # 计算聚合值
                agg_values = data.groupby(self.group_columns)[agg_col].transform(agg_func)
                result[new_col] = agg_values
                self._derived_columns.append(new_col)

        logger.info(f"Derived {len(self._derived_columns)} aggregation features")
        return result


class DateTimeDeriver(FeatureDeriverBase):
    """
    日期时间特征衍生器

    从日期时间列中提取特征。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 extract_features: List[str] = None):
        """
        初始化日期时间特征衍生器

        Args:
            columns: 日期时间列
            extract_features: 要提取的特征
                - year, month, day, hour, minute, second
                - dayofweek, dayofyear, weekofyear
                - quarter, is_weekend, is_month_start, is_month_end
        """
        super().__init__(columns)
        self.extract_features = extract_features or [
            'year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend'
        ]

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生日期时间特征"""
        logger.info("Deriving datetime features")

        result = data.copy()

        if self.columns:
            cols = self.columns
        else:
            # 自动检测日期时间列
            cols = data.select_dtypes(include=['datetime64']).columns.tolist()

        for col in cols:
            if col not in result.columns:
                continue

            try:
                dt_col = pd.to_datetime(result[col])
            except:
                logger.warning(f"Could not convert {col} to datetime")
                continue

            for feature in self.extract_features:
                new_col = f"{col}_{feature}"

                if feature == 'year':
                    result[new_col] = dt_col.dt.year
                elif feature == 'month':
                    result[new_col] = dt_col.dt.month
                elif feature == 'day':
                    result[new_col] = dt_col.dt.day
                elif feature == 'hour':
                    result[new_col] = dt_col.dt.hour
                elif feature == 'minute':
                    result[new_col] = dt_col.dt.minute
                elif feature == 'second':
                    result[new_col] = dt_col.dt.second
                elif feature == 'dayofweek':
                    result[new_col] = dt_col.dt.dayofweek
                elif feature == 'dayofyear':
                    result[new_col] = dt_col.dt.dayofyear
                elif feature == 'weekofyear':
                    result[new_col] = dt_col.dt.isocalendar().week
                elif feature == 'quarter':
                    result[new_col] = dt_col.dt.quarter
                elif feature == 'is_weekend':
                    result[new_col] = (dt_col.dt.dayofweek >= 5).astype(int)
                elif feature == 'is_month_start':
                    result[new_col] = dt_col.dt.is_month_start.astype(int)
                elif feature == 'is_month_end':
                    result[new_col] = dt_col.dt.is_month_end.astype(int)

                self._derived_columns.append(new_col)

        logger.info(f"Derived {len(self._derived_columns)} datetime features")
        return result


class MathDeriver(FeatureDeriverBase):
    """
    数学运算特征衍生器

    应用数学函数生成新特征。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 functions: List[str] = None):
        """
        初始化数学运算特征衍生器

        Args:
            columns: 要处理的列
            functions: 要应用的数学函数
                - log, log1p, log10, sqrt, square, abs
                - sin, cos, tan, exp, sign
        """
        super().__init__(columns)
        self.functions = functions or ['log1p', 'sqrt', 'square']

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生数学运算特征"""
        logger.info("Deriving math features")

        result = data.copy()

        if self.columns:
            cols = [c for c in self.columns if c in data.columns]
        else:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            col_data = data[col]

            for func in self.functions:
                new_col = f"{col}_{func}"

                try:
                    if func == 'log':
                        result[new_col] = np.log(col_data.clip(lower=1e-10))
                    elif func == 'log1p':
                        result[new_col] = np.log1p(col_data.clip(lower=0))
                    elif func == 'log10':
                        result[new_col] = np.log10(col_data.clip(lower=1e-10))
                    elif func == 'sqrt':
                        result[new_col] = np.sqrt(col_data.clip(lower=0))
                    elif func == 'square':
                        result[new_col] = col_data ** 2
                    elif func == 'abs':
                        result[new_col] = np.abs(col_data)
                    elif func == 'sin':
                        result[new_col] = np.sin(col_data)
                    elif func == 'cos':
                        result[new_col] = np.cos(col_data)
                    elif func == 'tan':
                        result[new_col] = np.tan(col_data)
                    elif func == 'exp':
                        result[new_col] = np.exp(col_data.clip(upper=700))  # 避免溢出
                    elif func == 'sign':
                        result[new_col] = np.sign(col_data)

                    self._derived_columns.append(new_col)
                except Exception as e:
                    logger.warning(f"Error applying {func} to {col}: {e}")

        logger.info(f"Derived {len(self._derived_columns)} math features")
        return result


class WindowDeriver(FeatureDeriverBase):
    """
    窗口特征衍生器

    计算滑动窗口统计特征。
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 window_sizes: List[int] = None,
                 functions: List[str] = None):
        """
        初始化窗口特征衍生器

        Args:
            columns: 要处理的列
            window_sizes: 窗口大小列表
            functions: 窗口函数（mean, sum, min, max, std）
        """
        super().__init__(columns)
        self.window_sizes = window_sizes or [3, 5, 7]
        self.functions = functions or ['mean', 'std']

    def derive(self, data: pd.DataFrame) -> pd.DataFrame:
        """衍生窗口特征"""
        logger.info("Deriving window features")

        result = data.copy()

        if self.columns:
            cols = [c for c in self.columns if c in data.columns]
        else:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            for window_size in self.window_sizes:
                for func in self.functions:
                    new_col = f"{col}_window{window_size}_{func}"

                    try:
                        window = result[col].rolling(window=window_size, min_periods=1)

                        if func == 'mean':
                            result[new_col] = window.mean()
                        elif func == 'sum':
                            result[new_col] = window.sum()
                        elif func == 'min':
                            result[new_col] = window.min()
                        elif func == 'max':
                            result[new_col] = window.max()
                        elif func == 'std':
                            result[new_col] = window.std()

                        self._derived_columns.append(new_col)
                    except Exception as e:
                        logger.warning(f"Error computing window {func} for {col}: {e}")

        logger.info(f"Derived {len(self._derived_columns)} window features")
        return result
