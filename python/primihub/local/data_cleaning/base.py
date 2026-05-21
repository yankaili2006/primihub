"""
Data Cleaning Base Classes
数据清洗基础类

提供各种数据清洗功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataCleanerBase(ABC):
    """数据清洗基类"""

    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据

        Args:
            data: 输入数据

        Returns:
            清洗后的数据
        """
        pass

    def get_report(self) -> Dict[str, Any]:
        """获取清洗报告"""
        return {}


class MissingValueHandler(DataCleanerBase):
    """
    缺失值处理器

    处理数据中的缺失值。
    """

    def __init__(self, strategy: str = "drop",
                 columns: Optional[List[str]] = None,
                 fill_value: Any = None,
                 fill_method: Optional[str] = None):
        """
        初始化缺失值处理器

        Args:
            strategy: 处理策略
                - drop: 删除含缺失值的行
                - fill: 填充缺失值
                - drop_columns: 删除缺失率超过阈值的列
            columns: 要处理的列，None表示所有列
            fill_value: 填充值（strategy='fill'时使用）
                - 数值：直接填充该值
                - 'mean': 使用均值填充
                - 'median': 使用中位数填充
                - 'mode': 使用众数填充
                - 'ffill': 前向填充
                - 'bfill': 后向填充
            fill_method: 填充方法（已弃用，使用fill_value）
        """
        self.strategy = strategy
        self.columns = columns
        self.fill_value = fill_value
        self.fill_method = fill_method
        self._report = {}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗缺失值"""
        logger.info(f"Handling missing values with strategy: {self.strategy}")

        result = data.copy()
        cols = self.columns if self.columns else result.columns.tolist()

        # 记录处理前的缺失情况
        before_missing = result[cols].isna().sum().sum()

        if self.strategy == "drop":
            result = result.dropna(subset=cols)
        elif self.strategy == "fill":
            result = self._fill_missing(result, cols)
        elif self.strategy == "drop_columns":
            # 删除缺失率超过50%的列
            missing_rates = result[cols].isna().mean()
            cols_to_drop = missing_rates[missing_rates > 0.5].index.tolist()
            result = result.drop(columns=cols_to_drop)
            self._report["dropped_columns"] = cols_to_drop

        # 记录处理后的缺失情况
        after_missing = result.isna().sum().sum()
        self._report["before_missing"] = int(before_missing)
        self._report["after_missing"] = int(after_missing)
        self._report["rows_before"] = len(data)
        self._report["rows_after"] = len(result)

        logger.info(f"Missing values: {before_missing} -> {after_missing}")
        return result

    def _fill_missing(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """填充缺失值"""
        result = data.copy()

        for col in cols:
            if col not in result.columns:
                continue

            if self.fill_value == "mean":
                fill_val = result[col].mean()
            elif self.fill_value == "median":
                fill_val = result[col].median()
            elif self.fill_value == "mode":
                mode_vals = result[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else None
            elif self.fill_value == "ffill":
                result[col] = result[col].ffill()
                continue
            elif self.fill_value == "bfill":
                result[col] = result[col].bfill()
                continue
            else:
                fill_val = self.fill_value

            if fill_val is not None:
                result[col] = result[col].fillna(fill_val)

        return result

    def get_report(self) -> Dict[str, Any]:
        return self._report


class DuplicateHandler(DataCleanerBase):
    """
    重复值处理器

    处理数据中的重复记录。
    """

    def __init__(self, subset: Optional[List[str]] = None,
                 keep: str = "first"):
        """
        初始化重复值处理器

        Args:
            subset: 用于判断重复的列，None表示所有列
            keep: 保留策略（first, last, False）
        """
        self.subset = subset
        self.keep = keep
        self._report = {}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗重复值"""
        logger.info(f"Handling duplicates with keep={self.keep}")

        before_count = len(data)
        duplicate_count = data.duplicated(subset=self.subset, keep=False).sum()

        result = data.drop_duplicates(subset=self.subset, keep=self.keep)
        after_count = len(result)

        self._report = {
            "rows_before": before_count,
            "rows_after": after_count,
            "duplicates_found": int(duplicate_count),
            "duplicates_removed": before_count - after_count,
        }

        logger.info(f"Removed {before_count - after_count} duplicate rows")
        return result

    def get_report(self) -> Dict[str, Any]:
        return self._report


class OutlierHandler(DataCleanerBase):
    """
    异常值处理器

    处理数据中的异常值。
    """

    def __init__(self, method: str = "iqr",
                 threshold: float = 1.5,
                 strategy: str = "clip",
                 columns: Optional[List[str]] = None):
        """
        初始化异常值处理器

        Args:
            method: 检测方法（iqr, zscore）
            threshold: 阈值
            strategy: 处理策略（clip, drop, replace_mean, replace_median）
            columns: 要处理的列
        """
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.columns = columns
        self._report = {}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗异常值"""
        logger.info(f"Handling outliers with method={self.method}, strategy={self.strategy}")

        result = data.copy()
        if self.columns:
            cols = [c for c in self.columns if c in result.columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        total_outliers = 0

        for col in cols:
            col_data = result[col].dropna()
            if len(col_data) == 0:
                continue

            if self.method == "iqr":
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.threshold * iqr
                upper = q3 + self.threshold * iqr
            else:  # zscore
                mean, std = col_data.mean(), col_data.std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std

            outlier_mask = (result[col] < lower) | (result[col] > upper)
            outlier_count = outlier_mask.sum()
            total_outliers += outlier_count

            if self.strategy == "clip":
                result[col] = result[col].clip(lower=lower, upper=upper)
            elif self.strategy == "drop":
                result = result[~outlier_mask]
            elif self.strategy == "replace_mean":
                result.loc[outlier_mask, col] = col_data.mean()
            elif self.strategy == "replace_median":
                result.loc[outlier_mask, col] = col_data.median()

        self._report = {
            "method": self.method,
            "strategy": self.strategy,
            "total_outliers_found": int(total_outliers),
            "rows_before": len(data),
            "rows_after": len(result),
        }

        logger.info(f"Found {total_outliers} outliers")
        return result

    def get_report(self) -> Dict[str, Any]:
        return self._report


class DataTypeConverter(DataCleanerBase):
    """
    数据类型转换器

    转换数据列的类型。
    """

    def __init__(self, type_mapping: Optional[Dict[str, str]] = None,
                 auto_convert: bool = False):
        """
        初始化数据类型转换器

        Args:
            type_mapping: 列名到目标类型的映射
            auto_convert: 是否自动推断并转换类型
        """
        self.type_mapping = type_mapping or {}
        self.auto_convert = auto_convert
        self._report = {}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        logger.info("Converting data types")

        result = data.copy()
        conversions = []

        # 手动指定的转换
        for col, dtype in self.type_mapping.items():
            if col in result.columns:
                try:
                    old_dtype = str(result[col].dtype)
                    result[col] = result[col].astype(dtype)
                    conversions.append({
                        "column": col,
                        "from": old_dtype,
                        "to": dtype,
                        "success": True,
                    })
                except Exception as e:
                    conversions.append({
                        "column": col,
                        "from": str(result[col].dtype),
                        "to": dtype,
                        "success": False,
                        "error": str(e),
                    })

        # 自动类型推断
        if self.auto_convert:
            for col in result.columns:
                if col in self.type_mapping:
                    continue

                old_dtype = str(result[col].dtype)

                # 尝试转换为数值
                if result[col].dtype == object:
                    try:
                        result[col] = pd.to_numeric(result[col], errors='ignore')
                        if result[col].dtype != object:
                            conversions.append({
                                "column": col,
                                "from": old_dtype,
                                "to": str(result[col].dtype),
                                "success": True,
                                "auto": True,
                            })
                    except:
                        pass

        self._report = {"conversions": conversions}
        logger.info(f"Performed {len(conversions)} type conversions")
        return result

    def get_report(self) -> Dict[str, Any]:
        return self._report


class ValueReplacer(DataCleanerBase):
    """
    值替换器

    替换数据中的特定值。
    """

    def __init__(self, replacements: Optional[Dict[str, Dict]] = None,
                 global_replacements: Optional[Dict[Any, Any]] = None):
        """
        初始化值替换器

        Args:
            replacements: 按列替换，格式为 {列名: {旧值: 新值}}
            global_replacements: 全局替换，格式为 {旧值: 新值}
        """
        self.replacements = replacements or {}
        self.global_replacements = global_replacements or {}
        self._report = {}

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """替换值"""
        logger.info("Replacing values")

        result = data.copy()
        replace_count = 0

        # 全局替换
        if self.global_replacements:
            for old_val, new_val in self.global_replacements.items():
                mask = result == old_val
                replace_count += mask.sum().sum()
                result = result.replace(old_val, new_val)

        # 按列替换
        for col, mapping in self.replacements.items():
            if col in result.columns:
                for old_val, new_val in mapping.items():
                    mask = result[col] == old_val
                    replace_count += mask.sum()
                    result[col] = result[col].replace(old_val, new_val)

        self._report = {"total_replacements": int(replace_count)}
        logger.info(f"Replaced {replace_count} values")
        return result

    def get_report(self) -> Dict[str, Any]:
        return self._report
