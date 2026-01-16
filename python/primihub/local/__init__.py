"""
PrimiHub Local (Single-Party) Processing Module
单方本地数据处理模块

提供单节点本地执行的数据处理功能，不涉及多方协作。

功能模块：
- data_statistics: 数据统计
- data_cleaning: 数据清洗
- data_scaling: 数据缩放
- feature_encoding: 特征编码
- feature_binning: 特征分箱
- feature_selection: 特征筛选
- feature_derivation: 特征衍生
- ml_lr: 机器学习逻辑回归
- ml_xgb: 机器学习XGBoost
- python_script: Python脚本处理
- sql_processing: SQL处理
- training_logger: 学习日志记录
- log_exporter: 学习日志导出
"""

from .base import LocalBaseModel, LocalProcessorBase

__all__ = [
    "LocalBaseModel",
    "LocalProcessorBase",
]
