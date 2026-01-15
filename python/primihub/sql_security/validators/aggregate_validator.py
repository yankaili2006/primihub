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
聚合算子校验器

校验聚合函数的安全性。
"""

from typing import List

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, AggregateCall, AggregateType


class AggregateValidator(BaseValidator):
    """聚合算子校验器

    校验聚合函数的安全性。

    检查点:
    1. COUNT是否满足k-匿名（结果>=k）
    2. SUM/AVG是否会导致属性推断
    3. MIN/MAX是否泄露边界值
    4. COUNT(DISTINCT)的风险
    5. 无GROUP BY时的聚合风险
    """

    def get_name(self) -> str:
        return "AggregateValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_aggregates():
            return issues

        # 是否有GROUP BY
        has_group_by = parsed_sql.has_group_by()

        for agg in parsed_sql.aggregates:
            issues.extend(self._validate_aggregate(agg, parsed_sql, has_group_by))

        return issues

    def _validate_aggregate(self, agg: AggregateCall, parsed_sql: ParsedSQL, has_group_by: bool) -> List[ValidationIssue]:
        """校验单个聚合函数"""
        issues = []

        # 根据聚合类型进行不同检查
        if agg.agg_type == AggregateType.COUNT:
            issues.extend(self._check_count(agg, parsed_sql, has_group_by))
        elif agg.agg_type == AggregateType.COUNT_DISTINCT:
            issues.extend(self._check_count_distinct(agg, parsed_sql, has_group_by))
        elif agg.agg_type in (AggregateType.SUM, AggregateType.AVG):
            issues.extend(self._check_sum_avg(agg, parsed_sql, has_group_by))
        elif agg.agg_type in (AggregateType.MIN, AggregateType.MAX):
            issues.extend(self._check_min_max(agg, parsed_sql, has_group_by))

        return issues

    def _check_count(self, agg: AggregateCall, parsed_sql: ParsedSQL, has_group_by: bool) -> List[ValidationIssue]:
        """检查COUNT函数"""
        issues = []

        # COUNT带有特定列时检查该列的安全性
        if agg.arguments:
            for col in agg.arguments:
                table_name = self.resolve_column_table(col, parsed_sql)
                field_meta = self.get_field_security(table_name, col.column_name)

                # k-匿名检查
                min_k = field_meta.min_aggregation_count

                if has_group_by:
                    # 有GROUP BY时，每个分组的COUNT可能小于k
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.INSUFFICIENT_K_ANONYMITY,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"COUNT({self.format_column_name(col)}) 与GROUP BY结合使用，需确保每个分组至少有{min_k}条记录",
                        sql_fragment=agg.raw_text,
                        remediation=f"添加HAVING COUNT(*) >= {min_k} 过滤小分组",
                        affected_fields=[col.column_name],
                        affected_tables=[table_name] if table_name else []
                    ))

        # COUNT(*)在无GROUP BY时通常安全
        if not agg.arguments and not has_group_by:
            # 全表COUNT通常是安全的
            pass

        return issues

    def _check_count_distinct(self, agg: AggregateCall, parsed_sql: ParsedSQL, has_group_by: bool) -> List[ValidationIssue]:
        """检查COUNT(DISTINCT)函数"""
        issues = []

        if not agg.arguments:
            return issues

        for col in agg.arguments:
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            # COUNT(DISTINCT)可能泄露唯一值信息
            if field_meta.requires_protection():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.COUNT_DISCLOSURE,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"COUNT(DISTINCT {self.format_column_name(col)}) 可能泄露字段的唯一值数量",
                    sql_fragment=agg.raw_text,
                    remediation="考虑是否真的需要精确的唯一值计数，可使用近似算法",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

            # 敏感字段的DISTINCT计数更危险
            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.COUNT_DISCLOSURE,
                    risk_level=RiskLevel.HIGH,
                    description=f"在{field_meta.security_level.name}级别字段上使用COUNT(DISTINCT)，可能泄露敏感信息",
                    sql_fragment=agg.raw_text,
                    remediation="敏感字段的唯一值计数需要使用差分隐私或MPC技术",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues

    def _check_sum_avg(self, agg: AggregateCall, parsed_sql: ParsedSQL, has_group_by: bool) -> List[ValidationIssue]:
        """检查SUM/AVG函数"""
        issues = []

        if not agg.arguments:
            return issues

        for col in agg.arguments:
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            agg_name = agg.agg_type.value

            # 检查字段安全级别
            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                    risk_level=RiskLevel.HIGH,
                    description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(col)} 上计算{agg_name}",
                    sql_fragment=agg.raw_text,
                    remediation=f"敏感字段的{agg_name}计算需要使用MPC或同态加密",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

            # 属性推断风险
            if has_group_by and field_meta.requires_protection():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.ATTRIBUTE_INFERENCE,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"分组{agg_name}({self.format_column_name(col)}) 可能导致属性推断攻击",
                    sql_fragment=agg.raw_text,
                    remediation="确保每个分组有足够多的记录，或添加噪声保护",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

            # AVG在小分组上的特殊风险
            if agg.agg_type == AggregateType.AVG and has_group_by:
                min_k = field_meta.min_aggregation_count
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                    risk_level=RiskLevel.LOW,
                    description=f"AVG({self.format_column_name(col)}) 在小分组上可能近似于单个值",
                    sql_fragment=agg.raw_text,
                    remediation=f"使用HAVING COUNT(*) >= {min_k} 确保分组足够大",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues

    def _check_min_max(self, agg: AggregateCall, parsed_sql: ParsedSQL, has_group_by: bool) -> List[ValidationIssue]:
        """检查MIN/MAX函数"""
        issues = []

        if not agg.arguments:
            return issues

        for col in agg.arguments:
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            agg_name = agg.agg_type.value

            # MIN/MAX直接暴露边界值
            if field_meta.requires_protection():
                risk_level = RiskLevel.MEDIUM
                if field_meta.requires_encryption():
                    risk_level = RiskLevel.HIGH

                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.BOUNDARY_VALUE_LEAKAGE,
                    risk_level=risk_level,
                    description=f"{agg_name}({self.format_column_name(col)}) 直接暴露字段的边界值",
                    sql_fragment=agg.raw_text,
                    remediation="考虑是否真的需要精确边界值，可使用分位数或范围代替",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

            # 敏感字段的MIN/MAX
            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.BOUNDARY_VALUE_LEAKAGE,
                    risk_level=RiskLevel.CRITICAL,
                    description=f"在{field_meta.security_level.name}级别字段上使用{agg_name}，将泄露实际的极值",
                    sql_fragment=agg.raw_text,
                    remediation="敏感字段的极值计算需要使用MPC技术，或完全避免此类查询",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

            # 带GROUP BY时更危险
            if has_group_by and field_meta.requires_protection():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.BOUNDARY_VALUE_LEAKAGE,
                    risk_level=RiskLevel.HIGH,
                    description=f"分组{agg_name}可能泄露每个分组中的极端个体",
                    sql_fragment=agg.raw_text,
                    remediation="小分组的MIN/MAX几乎等于暴露个体值，需特别注意",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues
