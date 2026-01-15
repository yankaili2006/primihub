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
字段保密属性定义和管理模块

定义联邦分析中字段的安全级别和访问控制属性。
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


class FieldSecurityLevel(IntEnum):
    """字段安全级别枚举

    定义四个安全级别，数值越大安全要求越高:
    - PUBLIC: 公开字段，可以任意操作
    - PROTECTED: 受保护字段，允许聚合但不允许直接暴露
    - PRIVATE: 私有字段，需要MPC/加密计算
    - SENSITIVE: 高度敏感字段，仅允许计数或存在性查询
    """
    PUBLIC = 0
    PROTECTED = 1
    PRIVATE = 2
    SENSITIVE = 3

    @classmethod
    def from_string(cls, level_str: str) -> 'FieldSecurityLevel':
        """从字符串转换为安全级别

        Args:
            level_str: 安全级别字符串，如 "PUBLIC", "PRIVATE" 等

        Returns:
            对应的 FieldSecurityLevel 枚举值

        Raises:
            ValueError: 如果字符串无法识别
        """
        level_map = {
            'PUBLIC': cls.PUBLIC,
            'PROTECTED': cls.PROTECTED,
            'PRIVATE': cls.PRIVATE,
            'SENSITIVE': cls.SENSITIVE
        }
        upper_str = level_str.upper()
        if upper_str not in level_map:
            raise ValueError(f"Unknown security level: {level_str}")
        return level_map[upper_str]


@dataclass
class FieldMeta:
    """字段元数据

    描述单个字段的安全属性和访问控制配置。

    Attributes:
        field_name: 字段名称
        table_name: 所属表名
        data_type: 数据类型 (如 INT, VARCHAR, FLOAT 等)
        security_level: 安全级别
        min_aggregation_count: k-匿名要求的最小聚合数量
        allow_filter: 是否允许在 WHERE 中过滤
        allow_join: 是否允许作为 JOIN 键
        allow_order_by: 是否允许排序
        allow_group_by: 是否允许分组
        owner_party: 数据所有方标识
        extra_policies: 附加安全策略
    """
    field_name: str = ""
    table_name: str = ""
    data_type: str = "VARCHAR"
    security_level: FieldSecurityLevel = FieldSecurityLevel.PUBLIC
    min_aggregation_count: int = 5
    allow_filter: bool = True
    allow_join: bool = True
    allow_order_by: bool = True
    allow_group_by: bool = True
    owner_party: str = "default"
    extra_policies: Dict[str, Any] = field(default_factory=dict)

    def is_public(self) -> bool:
        """检查字段是否为公开级别"""
        return self.security_level == FieldSecurityLevel.PUBLIC

    def is_protected(self) -> bool:
        """检查字段是否为受保护级别"""
        return self.security_level == FieldSecurityLevel.PROTECTED

    def is_private(self) -> bool:
        """检查字段是否为私有级别"""
        return self.security_level == FieldSecurityLevel.PRIVATE

    def is_sensitive(self) -> bool:
        """检查字段是否为敏感级别"""
        return self.security_level == FieldSecurityLevel.SENSITIVE

    def requires_protection(self) -> bool:
        """检查字段是否需要保护 (PROTECTED 及以上)"""
        return self.security_level >= FieldSecurityLevel.PROTECTED

    def requires_encryption(self) -> bool:
        """检查字段是否需要加密计算 (PRIVATE 及以上)"""
        return self.security_level >= FieldSecurityLevel.PRIVATE

    @classmethod
    def from_dict(cls, data: Dict[str, Any], table_name: str = "") -> 'FieldMeta':
        """从字典创建 FieldMeta

        Args:
            data: 包含字段配置的字典
            table_name: 表名

        Returns:
            FieldMeta 实例
        """
        security_level = data.get('security_level', 'PUBLIC')
        if isinstance(security_level, str):
            security_level = FieldSecurityLevel.from_string(security_level)
        elif isinstance(security_level, int):
            security_level = FieldSecurityLevel(security_level)

        return cls(
            field_name=data.get('name', data.get('field_name', '')),
            table_name=table_name,
            data_type=data.get('data_type', 'VARCHAR'),
            security_level=security_level,
            min_aggregation_count=data.get('min_aggregation_count', 5),
            allow_filter=data.get('allow_filter', True),
            allow_join=data.get('allow_join', True),
            allow_order_by=data.get('allow_order_by', True),
            allow_group_by=data.get('allow_group_by', True),
            owner_party=data.get('owner_party', 'default'),
            extra_policies=data.get('extra_policies', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'field_name': self.field_name,
            'table_name': self.table_name,
            'data_type': self.data_type,
            'security_level': self.security_level.name,
            'min_aggregation_count': self.min_aggregation_count,
            'allow_filter': self.allow_filter,
            'allow_join': self.allow_join,
            'allow_order_by': self.allow_order_by,
            'allow_group_by': self.allow_group_by,
            'owner_party': self.owner_party,
            'extra_policies': self.extra_policies
        }


@dataclass
class TableSecurityConfig:
    """表级安全配置

    描述整个表的安全属性和字段配置。

    Attributes:
        table_name: 表名
        owner_party: 数据所有方
        fields: 字段安全配置映射
        global_min_k: 全局 k-匿名阈值
        allow_cross_party_join: 是否允许跨方 JOIN
        allow_select_star: 是否允许 SELECT *
        max_result_rows: 最大结果行数限制
    """
    table_name: str = ""
    owner_party: str = "default"
    fields: Dict[str, FieldMeta] = field(default_factory=dict)
    global_min_k: int = 5
    allow_cross_party_join: bool = True
    allow_select_star: bool = False
    max_result_rows: int = 10000

    def get_field(self, field_name: str) -> Optional[FieldMeta]:
        """获取字段的安全配置

        Args:
            field_name: 字段名

        Returns:
            FieldMeta 或 None
        """
        return self.fields.get(field_name)

    def get_field_or_default(self, field_name: str) -> FieldMeta:
        """获取字段配置，如果不存在则返回默认配置

        对于未配置的字段，默认为 PROTECTED 级别以确保安全。

        Args:
            field_name: 字段名

        Returns:
            FieldMeta 实例
        """
        if field_name in self.fields:
            return self.fields[field_name]

        # 未配置的字段默认为 PROTECTED 级别
        return FieldMeta(
            field_name=field_name,
            table_name=self.table_name,
            security_level=FieldSecurityLevel.PROTECTED,
            min_aggregation_count=self.global_min_k,
            owner_party=self.owner_party
        )

    def register_field(self, field_meta: FieldMeta) -> None:
        """注册字段配置

        Args:
            field_meta: 字段元数据
        """
        field_meta.table_name = self.table_name
        self.fields[field_meta.field_name] = field_meta

    def get_public_fields(self) -> list:
        """获取所有公开字段"""
        return [f for f in self.fields.values() if f.is_public()]

    def get_protected_fields(self) -> list:
        """获取所有受保护字段"""
        return [f for f in self.fields.values() if f.requires_protection()]

    def get_private_fields(self) -> list:
        """获取所有私有字段"""
        return [f for f in self.fields.values() if f.requires_encryption()]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableSecurityConfig':
        """从字典创建 TableSecurityConfig

        Args:
            data: 包含表配置的字典

        Returns:
            TableSecurityConfig 实例
        """
        table_name = data.get('name', data.get('table_name', ''))
        owner_party = data.get('owner_party', 'default')

        config = cls(
            table_name=table_name,
            owner_party=owner_party,
            global_min_k=data.get('global_min_k', 5),
            allow_cross_party_join=data.get('allow_cross_party_join', True),
            allow_select_star=data.get('allow_select_star', False),
            max_result_rows=data.get('max_result_rows', 10000)
        )

        # 解析字段配置
        fields_data = data.get('fields', [])
        for field_data in fields_data:
            if isinstance(field_data, dict):
                field_meta = FieldMeta.from_dict(field_data, table_name)
                field_meta.owner_party = owner_party
                config.fields[field_meta.field_name] = field_meta

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'table_name': self.table_name,
            'owner_party': self.owner_party,
            'fields': [f.to_dict() for f in self.fields.values()],
            'global_min_k': self.global_min_k,
            'allow_cross_party_join': self.allow_cross_party_join,
            'allow_select_star': self.allow_select_star,
            'max_result_rows': self.max_result_rows
        }
