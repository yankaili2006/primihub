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
连接算子校验器

校验JOIN操作的安全性。
"""

from typing import List

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, JoinInfo, JoinType


class JoinValidator(BaseValidator):
    """连接算子校验器

    校验JOIN操作的安全性。

    检查点:
    1. 跨方JOIN是否被允许
    2. JOIN键的安全级别
    3. JOIN类型对隐私的影响（LEFT/RIGHT可能泄露存在性）
    4. 是否建议使用PSI
    """

    def get_name(self) -> str:
        return "JoinValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_joins():
            return issues

        # 检查每个JOIN
        for join in parsed_sql.joins:
            issues.extend(self._validate_join(join, parsed_sql))

        return issues

    def _validate_join(self, join: JoinInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """校验单个JOIN"""
        issues = []

        # 获取JOIN涉及的表
        right_table = join.right_table.table_name if join.right_table else ""

        # 确定左表（从FROM或之前的JOIN）
        left_tables = [t.table_name for t in parsed_sql.from_tables]
        if join.left_table:
            left_tables = [join.left_table.table_name]

        # 检查跨方JOIN
        issues.extend(self._check_cross_party_join(join, left_tables, right_table))

        # 检查JOIN键的安全级别
        issues.extend(self._check_join_key_security(join))

        # 检查JOIN类型的隐私影响
        issues.extend(self._check_join_type_privacy(join, left_tables, right_table))

        return issues

    def _check_cross_party_join(self, join: JoinInfo, left_tables: List[str], right_table: str) -> List[ValidationIssue]:
        """检查跨方JOIN"""
        issues = []

        if not right_table:
            return issues

        # 检查是否涉及不同方的数据
        right_owner = self.context.schema_manager.get_table_owner(right_table)

        for left_table in left_tables:
            left_owner = self.context.schema_manager.get_table_owner(left_table)

            if left_owner != right_owner:
                # 跨方JOIN
                if not self.context.policy.allow_cross_party_join:
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.CROSS_PARTY_JOIN_FORBIDDEN,
                        risk_level=RiskLevel.CRITICAL,
                        description=f"禁止跨方JOIN: {left_table}({left_owner}) JOIN {right_table}({right_owner})",
                        sql_fragment=join.raw_text,
                        remediation="跨方数据联合需要使用PSI(隐私集合交集)协议",
                        affected_tables=[left_table, right_table]
                    ))
                else:
                    # 允许但需要警告
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.UNSAFE_JOIN,
                        risk_level=RiskLevel.HIGH,
                        description=f"跨方JOIN可能泄露数据关联关系: {left_table}({left_owner}) JOIN {right_table}({right_owner})",
                        sql_fragment=join.raw_text,
                        remediation="建议使用PSI协议进行安全的跨方数据联合",
                        affected_tables=[left_table, right_table]
                    ))

        return issues

    def _check_join_key_security(self, join: JoinInfo) -> List[ValidationIssue]:
        """检查JOIN键的安全级别"""
        issues = []

        for left_col, right_col in join.on_columns:
            # 检查左侧JOIN键
            left_field_meta = self.get_field_security(left_col.table_name, left_col.column_name)
            if not left_field_meta.allow_join:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.JOIN_ON_SENSITIVE_FIELD,
                    risk_level=RiskLevel.HIGH,
                    description=f"字段 {self.format_column_name(left_col)} 不允许用于JOIN",
                    sql_fragment=join.raw_text,
                    remediation="该字段被配置为不可JOIN，请使用其他字段",
                    affected_fields=[left_col.column_name],
                    affected_tables=[left_col.table_name] if left_col.table_name else []
                ))

            if left_field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.JOIN_ON_SENSITIVE_FIELD,
                    risk_level=RiskLevel.HIGH,
                    description=f"在{left_field_meta.security_level.name}级别字段 {self.format_column_name(left_col)} 上进行JOIN",
                    sql_fragment=join.raw_text,
                    remediation="敏感字段JOIN需要使用PSI协议",
                    affected_fields=[left_col.column_name],
                    affected_tables=[left_col.table_name] if left_col.table_name else []
                ))

            # 检查右侧JOIN键
            right_field_meta = self.get_field_security(right_col.table_name, right_col.column_name)
            if not right_field_meta.allow_join:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.JOIN_ON_SENSITIVE_FIELD,
                    risk_level=RiskLevel.HIGH,
                    description=f"字段 {self.format_column_name(right_col)} 不允许用于JOIN",
                    sql_fragment=join.raw_text,
                    remediation="该字段被配置为不可JOIN，请使用其他字段",
                    affected_fields=[right_col.column_name],
                    affected_tables=[right_col.table_name] if right_col.table_name else []
                ))

            if right_field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.JOIN_ON_SENSITIVE_FIELD,
                    risk_level=RiskLevel.HIGH,
                    description=f"在{right_field_meta.security_level.name}级别字段 {self.format_column_name(right_col)} 上进行JOIN",
                    sql_fragment=join.raw_text,
                    remediation="敏感字段JOIN需要使用PSI协议",
                    affected_fields=[right_col.column_name],
                    affected_tables=[right_col.table_name] if right_col.table_name else []
                ))

        return issues

    def _check_join_type_privacy(self, join: JoinInfo, left_tables: List[str], right_table: str) -> List[ValidationIssue]:
        """检查JOIN类型的隐私影响"""
        issues = []

        # LEFT/RIGHT JOIN可能泄露数据存在性
        if join.join_type in (JoinType.LEFT, JoinType.RIGHT, JoinType.FULL):
            # 检查是否涉及敏感表
            all_tables = left_tables + ([right_table] if right_table else [])
            sensitive_tables = []

            for table in all_tables:
                table_config = self.context.schema_manager.get_table_config(table)
                if table_config:
                    # 如果表有敏感字段，则认为是敏感表
                    if any(f.requires_encryption() for f in table_config.fields.values()):
                        sensitive_tables.append(table)

            if sensitive_tables:
                join_type_name = join.join_type.value
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.JOIN_TYPE_LEAKAGE,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"{join_type_name} JOIN 可能通过NULL值泄露数据存在性",
                    sql_fragment=join.raw_text,
                    remediation="考虑使用INNER JOIN避免存在性泄露，或确保业务允许这种信息泄露",
                    affected_tables=sensitive_tables
                ))

        # CROSS JOIN产生笛卡尔积，可能导致数据爆炸
        if join.join_type == JoinType.CROSS:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.UNSAFE_JOIN,
                risk_level=RiskLevel.MEDIUM,
                description="CROSS JOIN 产生笛卡尔积，可能导致结果集过大",
                sql_fragment=join.raw_text,
                remediation="确认是否真的需要笛卡尔积，考虑添加JOIN条件",
                affected_tables=left_tables + ([right_table] if right_table else [])
            ))

        return issues

    def requires_psi(self, join: JoinInfo) -> bool:
        """判断是否需要使用PSI协议

        Args:
            join: JOIN信息

        Returns:
            是否建议使用PSI
        """
        # 跨方JOIN通常建议使用PSI
        if join.right_table:
            right_owner = self.context.schema_manager.get_table_owner(join.right_table.table_name)
            if join.left_table:
                left_owner = self.context.schema_manager.get_table_owner(join.left_table.table_name)
                if left_owner != right_owner:
                    return True

        # JOIN键是敏感字段时建议使用PSI
        for left_col, right_col in join.on_columns:
            left_meta = self.get_field_security(left_col.table_name, left_col.column_name)
            right_meta = self.get_field_security(right_col.table_name, right_col.column_name)
            if left_meta.requires_encryption() or right_meta.requires_encryption():
                return True

        return False
