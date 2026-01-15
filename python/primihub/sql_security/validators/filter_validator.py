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
筛选算子校验器

校验WHERE和HAVING子句的安全性。
"""

import re
from typing import List

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, Condition, ComparisonOp, ColumnRef


class FilterValidator(BaseValidator):
    """筛选算子校验器

    校验WHERE和HAVING子句中的条件安全性。

    检查点:
    1. 私有字段不能在WHERE中进行精确匹配
    2. 敏感字段不能用于过滤
    3. LIKE模式不能过于精确
    4. IN子句值数量限制
    5. 范围查询需要足够宽的范围
    """

    def get_name(self) -> str:
        return "FilterValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        # 保存当前表供后续使用
        self._current_tables = [t.table_name for t in parsed_sql.from_tables]

        # 校验WHERE条件
        for condition in parsed_sql.where_conditions:
            issues.extend(self._validate_condition(condition, parsed_sql, "WHERE"))

        # 校验HAVING条件
        for condition in parsed_sql.having_conditions:
            issues.extend(self._validate_condition(condition, parsed_sql, "HAVING"))

        # 如果有原始WHERE子句但没解析出条件，尝试从原始文本检测
        if parsed_sql.where_raw and not parsed_sql.where_conditions:
            issues.extend(self._validate_raw_where(parsed_sql.where_raw, parsed_sql))

        return issues

    def _validate_condition(self, condition: Condition, parsed_sql: ParsedSQL, clause_type: str) -> List[ValidationIssue]:
        """校验单个条件"""
        issues = []

        # 获取条件中涉及的列
        columns = condition.get_columns()

        for column in columns:
            table_name = self.resolve_column_table(column, parsed_sql)
            field_meta = self.get_field_security(table_name, column.column_name)

            # 检查字段是否允许过滤
            if not field_meta.allow_filter:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
                    risk_level=RiskLevel.HIGH,
                    description=f"字段 {self.format_column_name(column)} 不允许在{clause_type}中使用",
                    sql_fragment=condition.raw_text,
                    remediation="该字段被配置为不可过滤，请移除该条件或使用其他方式",
                    affected_fields=[column.column_name],
                    affected_tables=[table_name] if table_name else []
                ))
                continue

            # 根据操作符类型进行不同检查
            if condition.operator == ComparisonOp.EQ:
                issues.extend(self._check_exact_match(condition, column, field_meta, table_name))
            elif condition.operator == ComparisonOp.LIKE:
                issues.extend(self._check_like_pattern(condition, column, field_meta, table_name))
            elif condition.operator == ComparisonOp.IN:
                issues.extend(self._check_in_clause(condition, column, field_meta, table_name))
            elif condition.operator == ComparisonOp.BETWEEN:
                issues.extend(self._check_range_query(condition, column, field_meta, table_name))
            elif condition.operator in (ComparisonOp.LT, ComparisonOp.LE, ComparisonOp.GT, ComparisonOp.GE):
                issues.extend(self._check_comparison(condition, column, field_meta, table_name))

        return issues

    def _check_exact_match(self, condition: Condition, column: ColumnRef, field_meta, table_name: str) -> List[ValidationIssue]:
        """检查精确匹配"""
        issues = []

        # 私有或敏感字段不允许精确匹配
        if field_meta.requires_encryption():
            risk_level = self.get_risk_level_for_security_level(field_meta.security_level)
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.EXACT_MATCH_ON_PRIVATE,
                risk_level=risk_level,
                description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(column)} 上进行精确匹配，可能泄露具体值",
                sql_fragment=condition.raw_text,
                remediation="使用范围查询替代精确匹配，或使用PSI进行安全匹配",
                affected_fields=[column.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        return issues

    def _check_like_pattern(self, condition: Condition, column: ColumnRef, field_meta, table_name: str) -> List[ValidationIssue]:
        """检查LIKE模式"""
        issues = []

        pattern = str(condition.right_operand) if condition.right_operand else ""
        # 移除引号
        pattern = pattern.strip("'\"")

        # 检查模式是否过于精确
        min_length = self.context.policy.min_like_pattern_length

        # 计算有效字符数（排除通配符）
        effective_chars = pattern.replace('%', '').replace('_', '')

        if len(effective_chars) < min_length:
            # 模式太短可能导致过多匹配，这通常是安全的
            pass
        elif '%' not in pattern and '_' not in pattern:
            # 没有通配符，等同于精确匹配
            if field_meta.requires_protection():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.LIKE_PATTERN_TOO_SPECIFIC,
                    risk_level=RiskLevel.HIGH,
                    description=f"LIKE模式 '{pattern}' 过于精确（无通配符），等同于精确匹配",
                    sql_fragment=condition.raw_text,
                    remediation="添加通配符 % 以增加模糊性，如 '%{pattern}%'",
                    affected_fields=[column.column_name],
                    affected_tables=[table_name] if table_name else []
                ))
        elif not pattern.startswith('%') and not pattern.endswith('%'):
            # 前后都没有%，模式较精确
            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.LIKE_PATTERN_TOO_SPECIFIC,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"LIKE模式 '{pattern}' 较为精确，可能泄露敏感信息",
                    sql_fragment=condition.raw_text,
                    remediation="考虑使用更宽泛的模式，如 '%{pattern}%'",
                    affected_fields=[column.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        return issues

    def _check_in_clause(self, condition: Condition, column: ColumnRef, field_meta, table_name: str) -> List[ValidationIssue]:
        """检查IN子句"""
        issues = []

        # 解析IN子句中的值
        values_str = str(condition.right_operand) if condition.right_operand else ""
        values = [v.strip().strip("'\"") for v in values_str.split(',')]
        num_values = len(values)

        # 检查值数量
        max_values = self.context.policy.max_in_clause_values

        if num_values > max_values:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
                risk_level=RiskLevel.LOW,
                description=f"IN子句包含 {num_values} 个值，超过限制 {max_values}",
                sql_fragment=condition.raw_text,
                remediation="减少IN子句中的值数量，或使用子查询",
                affected_fields=[column.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        # 私有字段不允许IN查询（相当于多次精确匹配）
        if field_meta.requires_encryption() and num_values > 0:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.EXACT_MATCH_ON_PRIVATE,
                risk_level=RiskLevel.HIGH,
                description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(column)} 上使用IN子句，等同于多次精确匹配",
                sql_fragment=condition.raw_text,
                remediation="使用PSI进行安全的集合交集查询",
                affected_fields=[column.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        return issues

    def _check_range_query(self, condition: Condition, column: ColumnRef, field_meta, table_name: str) -> List[ValidationIssue]:
        """检查BETWEEN范围查询"""
        issues = []

        # 解析范围值
        if isinstance(condition.right_operand, tuple) and len(condition.right_operand) == 2:
            try:
                lower = float(str(condition.right_operand[0]).strip("'\""))
                upper = float(str(condition.right_operand[1]).strip("'\""))
                range_width = abs(upper - lower)

                min_width = self.context.policy.min_range_width

                if range_width < min_width:
                    risk_level = RiskLevel.MEDIUM
                    if field_meta.requires_encryption():
                        risk_level = RiskLevel.HIGH

                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.NARROW_RANGE_FILTER,
                        risk_level=risk_level,
                        description=f"BETWEEN范围 ({lower}, {upper}) 宽度为 {range_width}，小于最小要求 {min_width}",
                        sql_fragment=condition.raw_text,
                        remediation=f"扩大查询范围至少 {min_width}",
                        affected_fields=[column.column_name],
                        affected_tables=[table_name] if table_name else []
                    ))
            except (ValueError, TypeError):
                # 非数值范围，跳过宽度检查
                pass

        return issues

    def _check_comparison(self, condition: Condition, column: ColumnRef, field_meta, table_name: str) -> List[ValidationIssue]:
        """检查比较操作"""
        issues = []

        # 敏感字段的比较操作需要注意
        if field_meta.is_sensitive():
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
                risk_level=RiskLevel.HIGH,
                description=f"在SENSITIVE级别字段 {self.format_column_name(column)} 上进行比较操作",
                sql_fragment=condition.raw_text,
                remediation="敏感字段不应直接用于过滤条件",
                affected_fields=[column.column_name],
                affected_tables=[table_name] if table_name else []
            ))

        return issues

    def _validate_raw_where(self, where_raw: str, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """从原始WHERE文本检测问题"""
        issues = []

        # 检查是否包含敏感字段
        for table in parsed_sql.from_tables:
            table_config = self.context.schema_manager.get_table_config(table.table_name)
            if not table_config:
                continue

            for field_name, field_meta in table_config.fields.items():
                if not field_meta.allow_filter and re.search(rf'\b{field_name}\b', where_raw, re.IGNORECASE):
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
                        risk_level=RiskLevel.HIGH,
                        description=f"检测到不可过滤字段 {field_name} 出现在WHERE子句中",
                        sql_fragment=where_raw[:100],
                        remediation="请移除该字段的过滤条件",
                        affected_fields=[field_name],
                        affected_tables=[table.table_name]
                    ))

        return issues
