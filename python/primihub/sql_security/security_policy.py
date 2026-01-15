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
安全策略配置模块

管理全局安全策略和表级安全配置。
"""

import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from .field_security import (
    FieldSecurityLevel,
    FieldMeta,
    TableSecurityConfig
)


@dataclass
class SecurityPolicy:
    """全局安全策略配置

    定义联邦分析SQL校验的全局策略。

    Attributes:
        default_k_anonymity: 默认 k-匿名阈值
        allow_sensitive_in_where: 是否允许敏感字段出现在 WHERE 中
        allow_cross_party_join: 是否允许跨方 JOIN
        allow_window_functions: 是否允许窗口函数
        allow_correlated_subquery: 是否允许关联子查询
        allow_select_star: 是否允许 SELECT *
        max_result_rows: 最大结果行数
        strict_mode: 是否启用严格模式
        min_range_width: 范围查询的最小宽度
        max_in_clause_values: IN 子句最大值数量
        min_like_pattern_length: LIKE 模式最小长度
    """
    default_k_anonymity: int = 5
    allow_sensitive_in_where: bool = False
    allow_cross_party_join: bool = True
    allow_window_functions: bool = False
    allow_correlated_subquery: bool = False
    allow_select_star: bool = False
    max_result_rows: int = 10000
    strict_mode: bool = True
    min_range_width: float = 10.0
    max_in_clause_values: int = 100
    min_like_pattern_length: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityPolicy':
        """从字典创建"""
        return cls(
            default_k_anonymity=data.get('default_k_anonymity', 5),
            allow_sensitive_in_where=data.get('allow_sensitive_in_where', False),
            allow_cross_party_join=data.get('allow_cross_party_join', True),
            allow_window_functions=data.get('allow_window_functions', False),
            allow_correlated_subquery=data.get('allow_correlated_subquery', False),
            allow_select_star=data.get('allow_select_star', False),
            max_result_rows=data.get('max_result_rows', 10000),
            strict_mode=data.get('strict_mode', True),
            min_range_width=data.get('min_range_width', 10.0),
            max_in_clause_values=data.get('max_in_clause_values', 100),
            min_like_pattern_length=data.get('min_like_pattern_length', 3)
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'default_k_anonymity': self.default_k_anonymity,
            'allow_sensitive_in_where': self.allow_sensitive_in_where,
            'allow_cross_party_join': self.allow_cross_party_join,
            'allow_window_functions': self.allow_window_functions,
            'allow_correlated_subquery': self.allow_correlated_subquery,
            'allow_select_star': self.allow_select_star,
            'max_result_rows': self.max_result_rows,
            'strict_mode': self.strict_mode,
            'min_range_width': self.min_range_width,
            'max_in_clause_values': self.max_in_clause_values,
            'min_like_pattern_length': self.min_like_pattern_length
        }


class SchemaSecurityManager:
    """Schema安全管理器

    管理所有表和字段的安全配置。
    """

    def __init__(self):
        self._tables: Dict[str, TableSecurityConfig] = {}
        self._table_aliases: Dict[str, str] = {}  # alias -> table_name

    def register_table(self, config: TableSecurityConfig) -> None:
        """注册表安全配置

        Args:
            config: 表安全配置
        """
        self._tables[config.table_name.lower()] = config

    def register_alias(self, alias: str, table_name: str) -> None:
        """注册表别名

        Args:
            alias: 别名
            table_name: 实际表名
        """
        self._table_aliases[alias.lower()] = table_name.lower()

    def clear_aliases(self) -> None:
        """清除所有别名"""
        self._table_aliases.clear()

    def get_table_config(self, table_name: str) -> Optional[TableSecurityConfig]:
        """获取表安全配置

        Args:
            table_name: 表名或别名

        Returns:
            TableSecurityConfig 或 None
        """
        name_lower = table_name.lower()

        # 先检查别名
        if name_lower in self._table_aliases:
            name_lower = self._table_aliases[name_lower]

        return self._tables.get(name_lower)

    def get_field_security(self, table_name: str, field_name: str) -> Optional[FieldMeta]:
        """获取字段安全配置

        Args:
            table_name: 表名
            field_name: 字段名

        Returns:
            FieldMeta 或 None
        """
        table_config = self.get_table_config(table_name)
        if table_config:
            return table_config.get_field(field_name)
        return None

    def get_field_security_or_default(self, table_name: str, field_name: str) -> FieldMeta:
        """获取字段安全配置，如果不存在则返回默认配置

        Args:
            table_name: 表名
            field_name: 字段名

        Returns:
            FieldMeta 实例
        """
        table_config = self.get_table_config(table_name)
        if table_config:
            return table_config.get_field_or_default(field_name)

        # 表不存在时返回默认配置（PROTECTED级别）
        return FieldMeta(
            field_name=field_name,
            table_name=table_name,
            security_level=FieldSecurityLevel.PROTECTED
        )

    def resolve_table_name(self, name: str) -> str:
        """解析表名（处理别名）

        Args:
            name: 表名或别名

        Returns:
            实际表名
        """
        name_lower = name.lower()
        return self._table_aliases.get(name_lower, name_lower)

    def has_table(self, table_name: str) -> bool:
        """检查表是否已注册

        Args:
            table_name: 表名

        Returns:
            是否存在
        """
        name_lower = table_name.lower()
        if name_lower in self._table_aliases:
            name_lower = self._table_aliases[name_lower]
        return name_lower in self._tables

    def get_all_tables(self) -> List[str]:
        """获取所有已注册的表名"""
        return list(self._tables.keys())

    def get_table_owner(self, table_name: str) -> str:
        """获取表的所有方

        Args:
            table_name: 表名

        Returns:
            所有方标识
        """
        config = self.get_table_config(table_name)
        return config.owner_party if config else "unknown"

    def are_same_party(self, table1: str, table2: str) -> bool:
        """检查两个表是否属于同一方

        Args:
            table1: 第一个表名
            table2: 第二个表名

        Returns:
            是否同一方
        """
        owner1 = self.get_table_owner(table1)
        owner2 = self.get_table_owner(table2)
        return owner1 == owner2

    def load_from_yaml(self, yaml_path: str) -> None:
        """从YAML文件加载配置

        Args:
            yaml_path: YAML文件路径
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._load_config(config)

    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """从字典加载配置

        Args:
            config: 配置字典
        """
        self._load_config(config)

    def load_from_json(self, json_str: str) -> None:
        """从JSON字符串加载配置

        Args:
            json_str: JSON字符串
        """
        config = json.loads(json_str)
        self._load_config(config)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """内部加载配置

        Args:
            config: 配置字典
        """
        # 支持顶层 sql_security 包装
        if 'sql_security' in config:
            config = config['sql_security']

        # 加载表配置
        tables_config = config.get('tables', [])
        for table_data in tables_config:
            table_config = TableSecurityConfig.from_dict(table_data)
            self.register_table(table_config)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'tables': [t.to_dict() for t in self._tables.values()]
        }


@dataclass
class ValidationContext:
    """校验上下文

    包含校验过程中需要的所有上下文信息。

    Attributes:
        schema_manager: Schema安全管理器
        policy: 安全策略
        current_party: 当前执行方
        participating_parties: 参与方列表
    """
    schema_manager: SchemaSecurityManager
    policy: SecurityPolicy
    current_party: str = "default"
    participating_parties: List[str] = field(default_factory=list)

    def is_cross_party_query(self, tables: List[str]) -> bool:
        """检查查询是否涉及多方

        Args:
            tables: 表名列表

        Returns:
            是否跨方
        """
        if len(tables) <= 1:
            return False

        parties = set()
        for table in tables:
            owner = self.schema_manager.get_table_owner(table)
            parties.add(owner)

        return len(parties) > 1

    def get_table_party(self, table_name: str) -> str:
        """获取表所属方

        Args:
            table_name: 表名

        Returns:
            所属方
        """
        return self.schema_manager.get_table_owner(table_name)


class SecurityConfigLoader:
    """安全配置加载器

    提供便捷的配置加载方法。
    """

    @staticmethod
    def load(config_path: Optional[str] = None,
             config_dict: Optional[Dict[str, Any]] = None) -> tuple:
        """加载配置

        Args:
            config_path: 配置文件路径
            config_dict: 配置字典

        Returns:
            (SecurityPolicy, SchemaSecurityManager) 元组
        """
        policy = SecurityPolicy()
        schema_manager = SchemaSecurityManager()

        config = {}
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_dict:
            config = config_dict

        # 支持顶层 sql_security 包装
        if 'sql_security' in config:
            config = config['sql_security']

        # 加载全局策略
        if 'global_policy' in config:
            policy = SecurityPolicy.from_dict(config['global_policy'])

        # 加载表配置
        schema_manager.load_from_dict(config)

        return policy, schema_manager

    @staticmethod
    def create_context(config_path: Optional[str] = None,
                       config_dict: Optional[Dict[str, Any]] = None,
                       current_party: str = "default") -> ValidationContext:
        """创建校验上下文

        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
            current_party: 当前执行方

        Returns:
            ValidationContext 实例
        """
        policy, schema_manager = SecurityConfigLoader.load(config_path, config_dict)
        return ValidationContext(
            schema_manager=schema_manager,
            policy=policy,
            current_party=current_party
        )
