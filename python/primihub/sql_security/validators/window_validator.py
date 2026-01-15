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
窗口函数校验器

校验窗口函数的安全性。
"""

from typing import List, Set

from .base import BaseValidator
from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..sql_parser import ParsedSQL, WindowCall


class WindowFunctionValidator(BaseValidator):
    """窗口函数校验器

    校验窗口函数的安全性。

    检查点:
    1. 窗口函数是否被策略允许
    2. PARTITION BY的安全性
    3. ROW_NUMBER/RANK等排名函数风险
    4. LAG/LEAD邻近值泄露风险
    """

    # 排名类窗口函数
    RANKING_FUNCTIONS: Set[str] = {
        'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
        'PERCENT_RANK', 'CUME_DIST'
    }

    # 邻近值访问函数
    ADJACENT_FUNCTIONS: Set[str] = {
        'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'
    }

    def get_name(self) -> str:
        return "WindowFunctionValidator"

    def validate(self, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        issues = []

        if not parsed_sql.has_window_functions():
            return issues

        # 首先检查策略是否允许窗口函数
        if not self.context.policy.allow_window_functions:
            for window in parsed_sql.window_functions:
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.WINDOW_FUNCTION_FORBIDDEN,
                    risk_level=RiskLevel.HIGH,
                    description=f"安全策略禁止使用窗口函数: {window.function_name}",
                    sql_fragment=window.raw_text,
                    remediation="窗口函数可能泄露个体级别信息，请使用普通聚合函数替代"
                ))
            return issues

        # 检查每个窗口函数
        for window in parsed_sql.window_functions:
            issues.extend(self._validate_window_function(window, parsed_sql))

        return issues

    def _validate_window_function(self, window: WindowCall, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """校验单个窗口函数"""
        issues = []

        func_name = window.function_name.upper()

        # 检查PARTITION BY安全性
        issues.extend(self._check_partition_by(window, parsed_sql))

        # 根据函数类型进行特定检查
        if func_name in self.RANKING_FUNCTIONS:
            issues.extend(self._check_ranking_function(window, parsed_sql))
        elif func_name in self.ADJACENT_FUNCTIONS:
            issues.extend(self._check_adjacent_function(window, parsed_sql))
        else:
            # 其他窗口函数（如窗口聚合）
            issues.extend(self._check_window_aggregate(window, parsed_sql))

        return issues

    def _check_partition_by(self, window: WindowCall, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查PARTITION BY安全性"""
        issues = []

        if not window.partition_by:
            # 无PARTITION BY，整个结果集作为一个窗口
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.WINDOW_FUNCTION_RISK,
                risk_level=RiskLevel.LOW,
                description=f"窗口函数 {window.function_name} 无PARTITION BY，作用于整个结果集",
                sql_fragment=window.raw_text,
                remediation="考虑添加PARTITION BY以限制窗口范围"
            ))
            return issues

        # 检查分区字段
        for col in window.partition_by:
            table_name = self.resolve_column_table(col, parsed_sql)
            field_meta = self.get_field_security(table_name, col.column_name)

            if field_meta.requires_encryption():
                issues.append(self.create_issue(
                    issue_type=ValidationIssueType.WINDOW_FUNCTION_RISK,
                    risk_level=RiskLevel.HIGH,
                    description=f"在{field_meta.security_level.name}级别字段 {self.format_column_name(col)} 上进行窗口分区",
                    sql_fragment=window.raw_text,
                    remediation="敏感字段用于窗口分区会暴露字段的不同值",
                    affected_fields=[col.column_name],
                    affected_tables=[table_name] if table_name else []
                ))

        # 分区字段过多
        if len(window.partition_by) >= 3:
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.WINDOW_FUNCTION_RISK,
                risk_level=RiskLevel.MEDIUM,
                description=f"窗口分区包含{len(window.partition_by)}个字段，可能产生小分区",
                sql_fragment=window.raw_text,
                remediation="过多分区字段可能导致每个分区只有少量记录"
            ))

        return issues

    def _check_ranking_function(self, window: WindowCall, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查排名函数风险"""
        issues = []

        func_name = window.function_name.upper()

        # ROW_NUMBER给每行唯一编号
        if func_name == 'ROW_NUMBER':
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.RANKING_FUNCTION_RISK,
                risk_level=RiskLevel.MEDIUM,
                description="ROW_NUMBER()为每行分配唯一编号，可能用于追踪个体",
                sql_fragment=window.raw_text,
                remediation="如非必要，考虑使用RANK()或DENSE_RANK()替代"
            ))

        # RANK/DENSE_RANK泄露相对排名
        if func_name in ('RANK', 'DENSE_RANK'):
            # 检查ORDER BY中是否有敏感字段
            for col, _ in window.order_by:
                table_name = self.resolve_column_table(col, parsed_sql)
                field_meta = self.get_field_security(table_name, col.column_name)

                if field_meta.requires_protection():
                    issues.append(self.create_issue(
                        issue_type=ValidationIssueType.RANKING_FUNCTION_RISK,
                        risk_level=RiskLevel.HIGH,
                        description=f"{func_name}()基于敏感字段 {self.format_column_name(col)} 排名，泄露相对位置",
                        sql_fragment=window.raw_text,
                        remediation="排名函数会暴露敏感值的相对大小关系",
                        affected_fields=[col.column_name],
                        affected_tables=[table_name] if table_name else []
                    ))

        # NTILE可能创建小分组
        if func_name == 'NTILE':
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.RANKING_FUNCTION_RISK,
                risk_level=RiskLevel.MEDIUM,
                description="NTILE()将数据分成固定数量的桶，可能创建小分组",
                sql_fragment=window.raw_text,
                remediation="确保每个桶有足够多的记录"
            ))

        return issues

    def _check_adjacent_function(self, window: WindowCall, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查邻近值访问函数风险"""
        issues = []

        func_name = window.function_name.upper()

        # LAG/LEAD访问相邻行
        if func_name in ('LAG', 'LEAD'):
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.ADJACENT_VALUE_LEAKAGE,
                risk_level=RiskLevel.HIGH,
                description=f"{func_name}()访问相邻行的值，可能泄露其他个体的信息",
                sql_fragment=window.raw_text,
                remediation="LAG/LEAD函数会暴露相邻记录的值，在隐私场景中应避免使用"
            ))

        # FIRST_VALUE/LAST_VALUE暴露边界值
        if func_name in ('FIRST_VALUE', 'LAST_VALUE'):
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.ADJACENT_VALUE_LEAKAGE,
                risk_level=RiskLevel.MEDIUM,
                description=f"{func_name}()暴露分区的边界值",
                sql_fragment=window.raw_text,
                remediation="边界值函数可能泄露极端情况下的个体信息"
            ))

        # NTH_VALUE访问特定位置
        if func_name == 'NTH_VALUE':
            issues.append(self.create_issue(
                issue_type=ValidationIssueType.ADJACENT_VALUE_LEAKAGE,
                risk_level=RiskLevel.MEDIUM,
                description="NTH_VALUE()访问特定位置的值",
                sql_fragment=window.raw_text,
                remediation="访问特定位置可能暴露该位置个体的信息"
            ))

        return issues

    def _check_window_aggregate(self, window: WindowCall, parsed_sql: ParsedSQL) -> List[ValidationIssue]:
        """检查窗口聚合函数"""
        issues = []

        # 窗口聚合（如SUM() OVER）的风险较低
        # 但如果分区很小，仍然存在风险

        if not window.partition_by:
            # 全局窗口聚合通常是安全的
            return issues

        issues.append(self.create_issue(
            issue_type=ValidationIssueType.WINDOW_FUNCTION_RISK,
            risk_level=RiskLevel.LOW,
            description=f"窗口聚合函数 {window.function_name}() 在小分区上可能泄露信息",
            sql_fragment=window.raw_text,
            remediation="确保每个分区有足够多的记录"
        ))

        return issues
