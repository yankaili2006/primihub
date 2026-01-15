# Copyright 2024 PrimiHub
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PrimiHub SQL安全校验模块

提供联邦分析场景下的SQL安全校验功能，包括：
- 字段保密属性管理
- 各类SQL算子的安全检查
- k-匿名和隐私风险分析
- 函数安全校验（字符串、日期、时间戳、数值）
- SQL格式化和标准化

Usage:
    from primihub.sql_security import SQLSecurityEngine, validate_sql

    # 方式1: 快速校验
    result = validate_sql(
        "SELECT name, AVG(salary) FROM users GROUP BY name",
        config_path="config/sql_security_config.yaml"
    )
    print(result.summary())

    # 方式2: 使用引擎实例
    engine = SQLSecurityEngine()
    engine.load_config("config/sql_security_config.yaml")
    result = engine.validate("SELECT * FROM users WHERE salary > 50000")

    if not result.is_valid:
        for issue in result.issues:
            print(f"[{issue.risk_level.name}] {issue.description}")
            print(f"  建议: {issue.remediation}")

    # 方式3: SQL格式化
    formatted = engine.format_sql("select * from users where id=1")
    print(formatted)
"""

from .field_security import (
    FieldSecurityLevel,
    FieldMeta,
    TableSecurityConfig
)
from .validation_result import (
    RiskLevel,
    ValidationIssueType,
    ValidationIssue,
    ValidationResult
)
from .security_policy import (
    SecurityPolicy,
    SchemaSecurityManager,
    ValidationContext,
    SecurityConfigLoader
)
from .sql_parser import (
    SQLParser,
    ParsedSQL,
    ColumnRef,
    TableRef,
    JoinInfo,
    AggregateCall,
    WindowCall,
    SubqueryInfo
)
from .engine import SQLSecurityEngine, create_engine

# 函数校验模块
from .functions import (
    # 字符串函数
    StringFunctionType,
    StringFunctionExtractor,
    StringFunctionValidator,
    is_safe_string_function,
    # 日期函数
    DateFunctionType,
    DateFunctionExtractor,
    DateFunctionValidator,
    is_safe_date_function,
    # 时间戳函数
    TimestampFunctionType,
    TimestampFunctionExtractor,
    TimestampFunctionValidator,
    is_safe_timestamp_function,
    # 数值函数
    NumericFunctionType,
    NumericFunctionExtractor,
    NumericFunctionValidator,
    is_safe_numeric_function,
    # 统一接口
    FunctionExtractor,
    FunctionValidator,
    validate_functions,
)

# 格式化模块
from .formatter import (
    SQLFormatter,
    SQLNormalizer,
    SQLPrettifier,
    SQLCompactor,
    FormatOptions,
    FormatStyle,
    KeywordCase,
    IdentifierCase,
    format_sql,
    normalize_sql,
    compact_sql,
    prettify_sql,
    compare_sql,
)

__version__ = "1.1.0"
__author__ = "PrimiHub Team"

__all__ = [
    # 核心类
    'SQLSecurityEngine',
    'create_engine',

    # 字段安全
    'FieldSecurityLevel',
    'FieldMeta',
    'TableSecurityConfig',

    # 校验结果
    'RiskLevel',
    'ValidationIssueType',
    'ValidationIssue',
    'ValidationResult',

    # 安全策略
    'SecurityPolicy',
    'SchemaSecurityManager',
    'ValidationContext',
    'SecurityConfigLoader',

    # SQL解析
    'SQLParser',
    'ParsedSQL',
    'ColumnRef',
    'TableRef',
    'JoinInfo',
    'AggregateCall',
    'WindowCall',
    'SubqueryInfo',

    # 函数校验
    'StringFunctionType',
    'StringFunctionExtractor',
    'StringFunctionValidator',
    'is_safe_string_function',
    'DateFunctionType',
    'DateFunctionExtractor',
    'DateFunctionValidator',
    'is_safe_date_function',
    'TimestampFunctionType',
    'TimestampFunctionExtractor',
    'TimestampFunctionValidator',
    'is_safe_timestamp_function',
    'NumericFunctionType',
    'NumericFunctionExtractor',
    'NumericFunctionValidator',
    'is_safe_numeric_function',
    'FunctionExtractor',
    'FunctionValidator',
    'validate_functions',

    # SQL格式化
    'SQLFormatter',
    'SQLNormalizer',
    'SQLPrettifier',
    'SQLCompactor',
    'FormatOptions',
    'FormatStyle',
    'KeywordCase',
    'IdentifierCase',
    'format_sql',
    'normalize_sql',
    'compact_sql',
    'prettify_sql',
    'compare_sql',

    # 便捷函数
    'validate_sql',
    'is_sql_safe',
]


# 单例引擎，用于便捷函数
_default_engine: SQLSecurityEngine = None


def _get_default_engine(config_path: str = None) -> SQLSecurityEngine:
    """获取默认引擎"""
    global _default_engine
    if _default_engine is None or config_path:
        _default_engine = SQLSecurityEngine(config_path)
    return _default_engine


def validate_sql(sql: str,
                 config_path: str = None,
                 config_dict: dict = None,
                 current_party: str = "default") -> ValidationResult:
    """便捷的SQL校验函数

    Args:
        sql: 待校验的SQL语句
        config_path: 可选的配置文件路径
        config_dict: 可选的配置字典
        current_party: 当前执行方

    Returns:
        ValidationResult 校验结果

    Example:
        result = validate_sql(
            "SELECT * FROM users",
            config_path="config/sql_security_config.yaml"
        )
        print(result.summary())
    """
    engine = SQLSecurityEngine()
    if config_path:
        engine.load_config(config_path)
    elif config_dict:
        engine.load_config_from_dict(config_dict)
    return engine.validate(sql, current_party)


def is_sql_safe(sql: str,
                config_path: str = None,
                config_dict: dict = None,
                current_party: str = "default") -> bool:
    """快速检查SQL是否安全

    Args:
        sql: 待校验的SQL语句
        config_path: 可选的配置文件路径
        config_dict: 可选的配置字典
        current_party: 当前执行方

    Returns:
        是否安全

    Example:
        if is_sql_safe("SELECT id FROM users"):
            # 执行查询
            pass
    """
    result = validate_sql(sql, config_path, config_dict, current_party)
    return result.is_valid


def create_table_config(table_name: str,
                        owner_party: str = "default",
                        fields: dict = None,
                        global_min_k: int = 5) -> TableSecurityConfig:
    """便捷创建表安全配置

    Args:
        table_name: 表名
        owner_party: 所有方
        fields: 字段配置字典，格式如:
            {
                "field_name": {
                    "security_level": "PRIVATE",
                    "allow_filter": False
                }
            }
        global_min_k: 全局k-匿名阈值

    Returns:
        TableSecurityConfig 实例

    Example:
        config = create_table_config(
            "users",
            owner_party="party_a",
            fields={
                "id": {"security_level": "PUBLIC"},
                "name": {"security_level": "PRIVATE"},
                "salary": {"security_level": "PRIVATE", "min_aggregation_count": 10}
            }
        )
        engine.register_table(config)
    """
    config = TableSecurityConfig(
        table_name=table_name,
        owner_party=owner_party,
        global_min_k=global_min_k
    )

    if fields:
        for field_name, field_config in fields.items():
            field_meta = FieldMeta.from_dict({
                'name': field_name,
                **field_config
            }, table_name)
            field_meta.owner_party = owner_party
            config.fields[field_name] = field_meta

    return config
