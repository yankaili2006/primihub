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
子查询校验器

校验关联子查询和非关联子查询的安全性。
"""

from typing import List

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, SubqueryInfo


class SubqueryValidator(BaseValidator):
    """子查询校验器

    校验关联子查询和非关联子查询的安全性。

    关联子查询检查点:
    1. 关联子查询是否被允许
    2. EXISTS子查询的存在性泄露
    3. IN子查询的成员推断风险

    非关联子查询检查点:
    1. 递归校验子查询内容
    2. 标量子查询风险
    3. 子查询结果的安全性
    """

    def get_name(self) -> str:
        return "SubqueryValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_subqueries():
            return issues

        for subquery in parsed_sql.subqueries:
            if subquery.is_correlated:
                issues.extend(self._validate_correlated_subquery(subquery, parsed_sql))
            else:
                issues.extend(self._validate_uncorrelated_subquery(subquery, parsed_sql))

        return issues

    def _validate_correlated_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """校验关联子查询"""
        issues = []

        # 检查策略是否允许关联子查询
        if not self.context.policy.allow_correlated_subquery:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.CORRELATED_SUBQUERY_FORBIDDEN,
                risk_level=RiskLevel.HIGH,
                description="安全策略禁止使用关联子查询",
                sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                remediation="关联子查询可能逐行泄露信息，请使用JOIN或其他方式重写"
            ))
            return issues

        # 根据子查询类型进行检查
        if subquery.subquery_type == 'exists':
            issues.extend(self._check_exists_subquery(subquery, parsed_sql))
        elif subquery.subquery_type == 'in':
            issues.extend(self._check_in_subquery(subquery, parsed_sql))
        elif subquery.subquery_type == 'scalar':
            issues.extend(self._check_scalar_subquery(subquery, parsed_sql))

        # 检查关联列的安全性
        issues.extend(self._check_correlated_columns(subquery, parsed_sql))

        return issues

    def _validate_uncorrelated_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """校验非关联子查询"""
        issues = []

        # 递归校验子查询内容
        if subquery.nested_parsed:
            # 使用当前校验器递归校验
            # 这里简化处理，只检查一些基本问题
            nested = subquery.nested_parsed

            if nested.select_all:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.SUBQUERY_RISK,
                    risk_level=RiskLevel.MEDIUM,
                    description="子查询中使用SELECT *",
                    sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                    remediation="明确指定子查询需要的字段"
                ))

        # 根据子查询类型进行检查
        if subquery.subquery_type == 'scalar':
            issues.extend(self._check_scalar_subquery(subquery, parsed_sql))
        elif subquery.subquery_type == 'in':
            issues.extend(self._check_uncorrelated_in_subquery(subquery, parsed_sql))
        elif subquery.subquery_type == 'from':
            issues.extend(self._check_from_subquery(subquery, parsed_sql))

        return issues

    def _check_exists_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查EXISTS子查询"""
        issues = []

        # EXISTS子查询泄露存在性信息
        issues.append(self.create_issue(
            issue_type=ValidationIssueType.EXISTS_LEAKAGE,
            risk_level=RiskLevel.MEDIUM,
            description="EXISTS子查询泄露数据存在性",
            sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
            remediation="EXISTS会暴露是否存在满足条件的记录，在跨方查询中需特别注意"
        ))

        # 如果关联到敏感字段，风险更高
        for col in subquery.correlated_columns:
            table_name = col.table_name
            field_meta = self.get_field_security(table_name, col.column_name)

            if field_meta.requires_protection():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.EXISTS_LEAKAGE,
                    risk_level=RiskLevel.HIGH,
                    description=f"EXISTS子查询关联到受保护字段 {self.format_column_name(col)}",
                    sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                    remediation="通过EXISTS可以推断特定值是否存在于另一方数据中",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues

    def _check_in_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查关联IN子查询"""
        issues = []

        issues.append(self.create_issue(
            issue_type=ValidationIssueType.IN_SUBQUERY_MEMBERSHIP,
            risk_level=RiskLevel.MEDIUM,
            description="IN子查询可用于成员推断攻击",
            sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
            remediation="IN子查询会暴露值是否在某个集合中，建议使用PSI替代"
        ))

        return issues

    def _check_scalar_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查标量子查询"""
        issues = []

        issues.append(self.create_issue(
            issue_type=ValidationIssueType.SCALAR_SUBQUERY_RISK,
            risk_level=RiskLevel.MEDIUM,
            description="标量子查询返回单个值，可能泄露敏感信息",
            sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
            remediation="确保标量子查询不会返回敏感的聚合结果"
        ))

        # 检查子查询中是否有敏感聚合
        if subquery.nested_parsed and subquery.nested_parsed.aggregates:
            for agg in subquery.nested_parsed.aggregates:
                for col in agg.arguments:
                    table_name = col.table_name
                    field_meta = self.get_field_security(table_name, col.column_name)

                    if field_meta.requires_encryption():
                        issues.append(self.create_issue(
                            issue_type=ValidationIssueType.SCALAR_SUBQUERY_RISK,
                            risk_level=RiskLevel.HIGH,
                            description=f"标量子查询聚合敏感字段 {self.format_column_name(col)}",
                            sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                            remediation="敏感字段的聚合结果通过标量子查询返回，风险较高",
                            affected_fields=[col.column_name],
                            affected_tables=[table_name] if table_name else []
                        ))

        return issues

    def _check_uncorrelated_in_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查非关联IN子查询"""
        issues = []

        # 非关联IN子查询相对安全，但仍需检查
        if subquery.nested_parsed:
            nested = subquery.nested_parsed

            # 检查子查询返回的字段是否敏感
            for col in nested.select_columns:
                table_name = self.resolve_column_table(col, nested)
                field_meta = self.get_field_security(table_name, col.column_name)

                if field_meta.requires_encryption():
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.IN_SUBQUERY_MEMBERSHIP,
                        risk_level=RiskLevel.HIGH,
                        description=f"IN子查询返回敏感字段 {self.format_column_name(col)}",
                        sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                        remediation="敏感字段作为IN子查询的结果可能导致成员推断",
                        affected_fields=[col.column_name],
                        affected_tables=[table_name] if table_name else []
                    ))

        return issues

    def _check_from_subquery(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查FROM子句中的子查询"""
        issues = []

        # FROM子查询（派生表）通常风险较低
        issues.append(self.create_issue(
            issue_type=ValidationIssueType.SUBQUERY_RISK,
            risk_level=RiskLevel.LOW,
            description="FROM子句中的子查询（派生表）",
            sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
            remediation="确保派生表不会暴露敏感数据"
        ))

        return issues

    def _check_correlated_columns(self, subquery: SubqueryInfo, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查关联列的安全性"""
        issues = []

        for col in subquery.correlated_columns:
            table_name = col.table_name
            field_meta = self.get_field_security(table_name, col.column_name)

            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.SUBQUERY_RISK,
                    risk_level=RiskLevel.HIGH,
                    description=f"关联子查询使用敏感字段 {self.format_column_name(col)} 作为关联条件",
                    sql_fragment=subquery.raw_text[:100] if subquery.raw_text else "",
                    remediation="敏感字段作为关联条件会逐行暴露值的存在性",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues
