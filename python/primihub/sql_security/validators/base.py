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
校验器基类模块

定义所有校验器的抽象基类和通用方法。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..field_security import FieldMeta, FieldSecurityLevel
from ..validation_result import (
    ValidationIssue,
    ValidationIssueType,
    RiskLevel
)
from ..security_policy import ValidationContext
from ..sql_parser import ParsedSQL, ColumnRef


class BaseValidator(ABC):
    """校验器抽象基类

    所有具体校验器都应继承此类并实现 validate 方法。
    """

    def __init__(self, context: ValidationContext):
        """初始化校验器

        Args:
            context: 校验上下文
        """
        self.context = context

    @abstractmethod
    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """执行校验

        Args:
            parsed_sql: 解析后的SQL

        Returns:
            发现的问题列表
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取校验器名称

        Returns:
            校验器名称
        """
        pass

    def create_issue(self,
                     issue_type: ValidationIssueType,
                     risk_level: RiskLevel,
                     description: str,
                     sql_fragment: str = "",
                     remediation: str = "",
                     affected_fields: Optional[List[str]] = None,
                     affected_tables: Optional[List[str]] = None) -> ValidationIssue:
        """创建校验问题

        Args:
            issue_type: 问题类型
            risk_level: 风险等级
            description: 问题描述
            sql_fragment: 相关SQL片段
            remediation: 建议修复方案
            affected_fields: 受影响的字段
            affected_tables: 受影响的表

        Returns:
            ValidationIssue 实例
        """
        return ValidationIssue(
            issue_type=issue_type,
            risk_level=risk_level,
            description=description,
            sql_fragment=sql_fragment,
            remediation=remediation,
            affected_fields=affected_fields or [],
            affected_tables=affected_tables or [],
            validator_name=self.get_name()
        )

    def get_field_security(self, table_name: str, field_name: str) -> FieldMeta:
        """获取字段安全配置

        Args:
            table_name: 表名
            field_name: 字段名

        Returns:
            FieldMeta 实例
        """
        return self.context.schema_manager.get_field_security_or_default(
            table_name, field_name
        )

    def get_column_field_meta(self, column: ColumnRef) -> FieldMeta:
        """获取列引用对应的字段安全配置

        Args:
            column: 列引用

        Returns:
            FieldMeta 实例
        """
        table_name = column.table_name
        # 如果没有表名，尝试从上下文推断
        if not table_name and hasattr(self, '_current_tables'):
            if len(self._current_tables) == 1:
                table_name = self._current_tables[0]

        return self.get_field_security(table_name, column.column_name)

    def check_field_requires_protection(self, table_name: str, field_name: str) -> bool:
        """检查字段是否需要保护

        Args:
            table_name: 表名
            field_name: 字段名

        Returns:
            是否需要保护
        """
        field_meta = self.get_field_security(table_name, field_name)
        return field_meta.requires_protection()

    def check_field_requires_encryption(self, table_name: str, field_name: str) -> bool:
        """检查字段是否需要加密

        Args:
            table_name: 表名
            field_name: 字段名

        Returns:
            是否需要加密
        """
        field_meta = self.get_field_security(table_name, field_name)
        return field_meta.requires_encryption()

    def resolve_column_table(self, column: ColumnRef, parsed_sql: ParsedSQL) -> str:
        """解析列所属的表

        Args:
            column: 列引用
            parsed_sql: 解析后的SQL

        Returns:
            表名
        """
        if column.table_name:
            # 检查是否是别名
            resolved = self.context.schema_manager.resolve_table_name(column.table_name)
            return resolved

        # 如果没有表名，尝试从FROM表推断
        if len(parsed_sql.from_tables) == 1:
            return parsed_sql.from_tables[0].table_name

        # 多个表时，搜索包含该字段的表
        for table in parsed_sql.from_tables:
            table_config = self.context.schema_manager.get_table_config(table.table_name)
            if table_config and column.column_name in table_config.fields:
                return table.table_name

        return ""

    def get_risk_level_for_security_level(self, security_level: FieldSecurityLevel) -> RiskLevel:
        """根据字段安全级别获取对应的风险等级

        Args:
            security_level: 字段安全级别

        Returns:
            风险等级
        """
        mapping = {
            FieldSecurityLevel.PUBLIC: RiskLevel.SAFE,
            FieldSecurityLevel.PROTECTED: RiskLevel.MEDIUM,
            FieldSecurityLevel.PRIVATE: RiskLevel.HIGH,
            FieldSecurityLevel.SENSITIVE: RiskLevel.CRITICAL
        }
        return mapping.get(security_level, RiskLevel.MEDIUM)

    def format_column_name(self, column: ColumnRef) -> str:
        """格式化列名用于显示

        Args:
            column: 列引用

        Returns:
            格式化后的列名
        """
        if column.table_name:
            return f"{column.table_name}.{column.column_name}"
        return column.column_name

    def format_columns(self, columns: List[ColumnRef]) -> str:
        """格式化多个列名

        Args:
            columns: 列引用列表

        Returns:
            格式化后的字符串
        """
        return ", ".join(self.format_column_name(c) for c in columns)


class SelectValidator(BaseValidator):
    """SELECT列校验器

    校验SELECT子句中的字段访问安全性。
    """

    def get_name(self) -> str:
        return "SelectValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        # 检查SELECT *
        if parsed_sql.select_all:
            if not self.context.policy.allow_select_star:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.SELECT_STAR_FORBIDDEN,
                    risk_level=RiskLevel.HIGH,
                    description="禁止使用 SELECT *，可能暴露敏感字段",
                    sql_fragment="SELECT *",
                    remediation="请明确指定需要查询的字段"
                ))
            else:
                # 检查是否会暴露敏感字段
                for table in parsed_sql.from_tables:
                    table_config = self.context.schema_manager.get_table_config(table.table_name)
                    if table_config:
                        sensitive_fields = [
                            f.field_name for f in table_config.fields.values()
                            if f.requires_encryption()
                        ]
                        if sensitive_fields:
                            issues.append(self.create_issue(
                                issue_type=ValidationIssueType.DIRECT_FIELD_EXPOSURE,
                                risk_level=RiskLevel.HIGH,
                                description=f"SELECT * 将暴露敏感字段: {', '.join(sensitive_fields)}",
                                sql_fragment="SELECT *",
                                remediation="请排除敏感字段或使用加密计算",
                                affected_fields=sensitive_fields,
                                affected_tables=[table.table_name]
                            ))

        # 检查每个SELECT列
        for column in parsed_sql.select_columns:
            if column.column_name == '*':
                continue

            table_name = self.resolve_column_table(column, parsed_sql)
            field_meta = self.get_field_security(table_name, column.column_name)

            # PRIVATE或SENSITIVE字段直接暴露
            if field_meta.requires_encryption():
                risk_level = self.get_risk_level_for_security_level(field_meta.security_level)
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.DIRECT_FIELD_EXPOSURE,
                    risk_level=risk_level,
                    description=f"直接暴露{field_meta.security_level.name}级别字段: {self.format_column_name(column)}",
                    sql_fragment=self.format_column_name(column),
                    remediation="使用聚合函数或加密计算来保护该字段",
                    affected_fields=[column.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues
