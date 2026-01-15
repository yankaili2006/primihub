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
排序算子校验器

校验ORDER BY的安全性。
"""

from typing import List

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, OrderByItem


class OrderByValidator(BaseValidator):
    """排序算子校验器

    校验ORDER BY的安全性。

    检查点:
    1. 排序字段是否为敏感字段
    2. ORDER BY + LIMIT组合的TOP-N风险
    3. 排序是否可能推断敏感值排名
    """

    def get_name(self) -> str:
        return "OrderByValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_order_by():
            return issues

        # 检查每个排序字段
        for item in parsed_sql.order_by_items:
            issues.extend(self._check_order_by_field(item, parsed_sql))

        # 检查TOP-N风险
        if parsed_sql.limit is not None:
            issues.extend(self._check_top_n_risk(parsed_sql))

        return issues

    def _check_order_by_field(self, item: OrderByItem, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查单个排序字段"""
        issues = []

        col = item.column
        table_name = self.resolve_column_table(col, parsed_sql)
        field_meta = self.get_field_security(table_name, col.column_name)

        # 检查是否允许排序
        if not field_meta.allow_order_by:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.ORDER_BY_SENSITIVE,
                risk_level=RiskLevel.HIGH,
                description=f"字段 {self.format_column_name(col)} 不允许用于排序",
                sql_fragment=f"ORDER BY {self.format_column_name(col)}",
                remediation="该字段被配置为不可排序，请移除排序或使用其他字段",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        # 敏感字段排序
        if field_meta.requires_encryption():
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.ORDER_BY_SENSITIVE,
                risk_level=RiskLevel.HIGH,
                description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(col)} 上进行排序",
                sql_fragment=f"ORDER BY {self.format_column_name(col)}",
                remediation="敏感字段排序会暴露值的相对大小关系",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))
        elif field_meta.requires_protection():
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.PRIVACY_LEAKAGE_VIA_ORDER,
                risk_level=RiskLevel.MEDIUM,
                description=f"在PROTECTED级别字段 {self.format_column_name(col)} 上进行排序",
                sql_fragment=f"ORDER BY {self.format_column_name(col)}",
                remediation="排序可能泄露数据分布信息",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        return issues

    def _check_top_n_risk(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查TOP-N风险"""
        issues = []

        limit = parsed_sql.limit
        order_by_items = parsed_sql.order_by_items

        if not order_by_items:
            return issues

        # 小LIMIT值更危险
        if limit <= 10:
            risk_level = RiskLevel.HIGH
            description = f"ORDER BY配合LIMIT {limit}可能泄露极端值对应的记录"
        elif limit <= 100:
            risk_level = RiskLevel.MEDIUM
            description = f"ORDER BY配合LIMIT {limit}可能泄露排名靠前/后的记录"
        else:
            risk_level = RiskLevel.LOW
            description = f"ORDER BY配合LIMIT {limit}"

        # 检查排序字段是否敏感
        sensitive_fields = []
        for item in order_by_items:
            col = item.column
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            if field_meta.requires_protection():
                sensitive_fields.append(col.column_name)

        if sensitive_fields:
            direction_desc = "最大" if not order_by_items[0].is_ascending else "最小"
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.TOP_N_LEAKAGE,
                risk_level=risk_level,
                description=f"{description}，涉及敏感字段 {', '.join(sensitive_fields)}，可能暴露{direction_desc}值的个体",
                sql_fragment=f"ORDER BY ... LIMIT {limit}",
                remediation="考虑增大LIMIT值，或移除对敏感字段的排序",
                affected_fields=sensitive_fields
            ))

        # 即使字段不敏感，小LIMIT也需要警告
        if limit <= 5:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.TOP_N_LEAKAGE,
                risk_level=RiskLevel.MEDIUM,
                description=f"LIMIT {limit}结果集很小，可能导致个体识别",
                sql_fragment=f"ORDER BY ... LIMIT {limit}",
                remediation="考虑增大LIMIT值以降低个体识别风险"
            ))

        return issues
