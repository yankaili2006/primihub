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
分组算子校验器

校验GROUP BY的安全性。
"""

from typing import List, Set

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, ColumnRef


class GroupByValidator(BaseValidator):
    """分组算子校验器

    校验GROUP BY的安全性。

    检查点:
    1. 分组字段的安全级别
    2. 分组粒度是否过细
    3. 小分组泄露风险（组内记录数<k）
    4. 准标识符组合检测
    """

    # 常见的准标识符字段名
    QUASI_IDENTIFIER_PATTERNS = {
        'age', 'gender', 'sex', 'birth', 'zip', 'postal', 'city', 'region',
        'province', 'state', 'country', 'occupation', 'job', 'education',
        'marital', 'race', 'ethnicity', 'nationality'
    }

    def get_name(self) -> str:
        return "GroupByValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_group_by():
            return issues

        group_by_columns = parsed_sql.group_by_columns

        # 检查每个分组字段
        for col in group_by_columns:
            issues.extend(self._check_group_by_field(col, parsed_sql))

        # 检查分组粒度
        issues.extend(self._check_granularity(group_by_columns, parsed_sql))

        # 检查准标识符组合
        issues.extend(self._check_quasi_identifiers(group_by_columns, parsed_sql))

        return issues

    def _check_group_by_field(self, col: ColumnRef, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查单个分组字段"""
        issues = []

        table_name = self.resolve_column_table(col, parsed_sql)
        field_meta = self.get_field_security(table_name, col.column_name)

        # 检查是否允许分组
        if not field_meta.allow_group_by:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.GROUP_BY_ON_SENSITIVE,
                risk_level=RiskLevel.HIGH,
                description=f"字段 {self.format_column_name(col)} 不允许用于GROUP BY",
                sql_fragment=f"GROUP BY {self.format_column_name(col)}",
                remediation="该字段被配置为不可分组，请使用其他字段",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        # 敏感字段用于分组
        if field_meta.requires_encryption():
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.GROUP_BY_ON_SENSITIVE,
                risk_level=RiskLevel.HIGH,
                description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(col)} 上进行分组",
                sql_fragment=f"GROUP BY {self.format_column_name(col)}",
                remediation="敏感字段分组会暴露字段的所有不同值，需使用泛化或加密技术",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))
        elif field_meta.requires_protection():
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.GROUP_BY_ON_SENSITIVE,
                risk_level=RiskLevel.MEDIUM,
                description=f"在PROTECTED级别字段 {self.format_column_name(col)} 上进行分组",
                sql_fragment=f"GROUP BY {self.format_column_name(col)}",
                remediation="建议对分组字段进行泛化处理（如年龄分段）",
                affected_fields=[col.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        return issues

    def _check_granularity(self, columns: List[ColumnRef], parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查分组粒度"""
        issues = []

        # 分组字段越多，粒度越细
        num_columns = len(columns)

        if num_columns >= 3:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.FINE_GRAINED_GROUPING,
                risk_level=RiskLevel.MEDIUM,
                description=f"GROUP BY包含{num_columns}个字段，分组粒度可能过细",
                sql_fragment=f"GROUP BY {self.format_columns(columns)}",
                remediation="减少分组字段数量，或确保每个分组有足够的记录",
                affected_fields=[c.column_name for c in columns]
            ))

        if num_columns >= 5:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.FINE_GRAINED_GROUPING,
                risk_level=RiskLevel.HIGH,
                description=f"GROUP BY包含{num_columns}个字段，很可能产生大量小分组",
                sql_fragment=f"GROUP BY {self.format_columns(columns)}",
                remediation="强烈建议减少分组字段，过细的分组可能导致个体识别",
                affected_fields=[c.column_name for c in columns]
            ))

        # 检查是否有高基数字段
        for col in columns:
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            # 检查是否是高基数字段（如ID类）
            col_lower = col.column_name.lower()
            if any(pattern in col_lower for pattern in ['id', 'code', 'key', 'number', 'no']):
                if not col_lower.endswith('_type') and not col_lower.endswith('_category'):
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.FINE_GRAINED_GROUPING,
                        risk_level=RiskLevel.HIGH,
                        description=f"字段 {self.format_column_name(col)} 可能是高基数字段，不适合用于分组",
                        sql_fragment=f"GROUP BY {self.format_column_name(col)}",
                        remediation="ID类字段用于分组可能导致每个分组只有一条记录",
                        affected_fields=[col.column_name],
                        affected_tables=[table_name] if table_name else []
                    ))

        return issues

    def _check_quasi_identifiers(self, columns: List[ColumnRef], parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查准标识符组合"""
        issues = []

        # 识别可能的准标识符
        quasi_identifiers: Set[str] = set()
        for col in columns:
            col_lower = col.column_name.lower()
            for pattern in self.QUASI_IDENTIFIER_PATTERNS:
                if pattern in col_lower:
                    quasi_identifiers.add(col.column_name)
                    break

        # 多个准标识符组合增加重识别风险
        if len(quasi_identifiers) >= 2:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.QUASI_IDENTIFIER_RISK,
                risk_level=RiskLevel.HIGH,
                description=f"分组包含多个准标识符字段: {', '.join(quasi_identifiers)}，可能导致重识别攻击",
                sql_fragment=f"GROUP BY {self.format_columns(columns)}",
                remediation="准标识符组合（如年龄+性别+邮编）可能唯一标识个体，建议减少或泛化",
                affected_fields=list(quasi_identifiers)
            ))

        if len(quasi_identifiers) >= 3:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.QUASI_IDENTIFIER_RISK,
                risk_level=RiskLevel.CRITICAL,
                description=f"分组包含{len(quasi_identifiers)}个准标识符，重识别风险极高",
                sql_fragment=f"GROUP BY {self.format_columns(columns)}",
                remediation="研究表明3个以上准标识符组合足以唯一识别大多数个体",
                affected_fields=list(quasi_identifiers)
            ))

        return issues

    def _check_small_group_risk(self, columns: List[ColumnRef], parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查小分组风险

        注：这是静态检查，无法知道实际分组大小，
        只能基于字段特性给出警告。
        """
        issues = []

        # 获取k-匿名要求
        min_k = self.context.policy.default_k_anonymity

        # 如果没有HAVING子句限制分组大小，给出警告
        if not parsed_sql.having_conditions and not parsed_sql.having_raw:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.SMALL_GROUP_RISK,
                risk_level=RiskLevel.LOW,
                description=f"GROUP BY查询没有HAVING子句限制分组大小",
                sql_fragment=f"GROUP BY {self.format_columns(columns)}",
                remediation=f"建议添加 HAVING COUNT(*) >= {min_k} 以满足k-匿名要求"
            ))

        return issues
