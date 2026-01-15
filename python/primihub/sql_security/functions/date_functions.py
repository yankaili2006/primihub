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
日期类型函数模块

提供联邦分析场景下日期函数的安全校验和处理。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..field_security import FieldMeta, FieldSecurityLevel


class DateFunctionType(Enum):
    """日期函数类型"""
    # 当前日期/时间
    CURRENT_DATE = "CURRENT_DATE"
    CURDATE = "CURDATE"
    CURRENT_TIME = "CURRENT_TIME"
    CURTIME = "CURTIME"
    NOW = "NOW"
    SYSDATE = "SYSDATE"
    GETDATE = "GETDATE"

    # 日期提取
    YEAR = "YEAR"
    MONTH = "MONTH"
    DAY = "DAY"
    DAYOFWEEK = "DAYOFWEEK"
    DAYOFMONTH = "DAYOFMONTH"
    DAYOFYEAR = "DAYOFYEAR"
    WEEK = "WEEK"
    WEEKOFYEAR = "WEEKOFYEAR"
    QUARTER = "QUARTER"
    HOUR = "HOUR"
    MINUTE = "MINUTE"
    SECOND = "SECOND"

    # 日期提取 - EXTRACT函数
    EXTRACT = "EXTRACT"

    # 日期格式化
    DATE_FORMAT = "DATE_FORMAT"
    TIME_FORMAT = "TIME_FORMAT"
    TO_CHAR = "TO_CHAR"
    FORMAT = "FORMAT"

    # 日期解析
    STR_TO_DATE = "STR_TO_DATE"
    TO_DATE = "TO_DATE"
    PARSE_DATE = "PARSE_DATE"

    # 日期运算
    DATE_ADD = "DATE_ADD"
    DATE_SUB = "DATE_SUB"
    ADDDATE = "ADDDATE"
    SUBDATE = "SUBDATE"
    DATEDIFF = "DATEDIFF"
    TIMESTAMPDIFF = "TIMESTAMPDIFF"
    DATE_DIFF = "DATE_DIFF"

    # 日期截断
    DATE_TRUNC = "DATE_TRUNC"
    TRUNC = "TRUNC"
    TRUNCATE = "TRUNCATE"

    # 日期构造
    MAKEDATE = "MAKEDATE"
    MAKETIME = "MAKETIME"
    DATE = "DATE"
    TIME = "TIME"

    # 日期边界
    LAST_DAY = "LAST_DAY"
    FIRST_DAY = "FIRST_DAY"
    EOMONTH = "EOMONTH"

    # 日期比较
    DATE_CMP = "DATE_CMP"

    # 其他
    AGE = "AGE"
    MONTHS_BETWEEN = "MONTHS_BETWEEN"
    ADD_MONTHS = "ADD_MONTHS"

    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, func_name: str) -> 'DateFunctionType':
        """从函数名转换"""
        name_upper = func_name.upper()
        for member in cls:
            if member.value == name_upper:
                return member
        return cls.UNKNOWN


@dataclass
class DateFunctionCall:
    """日期函数调用"""
    func_type: DateFunctionType
    func_name: str
    arguments: List[Any] = field(default_factory=list)
    raw_text: str = ""

    # 涉及的列
    column_refs: List[str] = field(default_factory=list)

    # 提取的时间单位 (用于EXTRACT, DATE_TRUNC等)
    time_unit: Optional[str] = None

    # 是否包含字面值
    has_literal: bool = False
    literal_values: List[str] = field(default_factory=list)


class DateFunctionExtractor:
    """日期函数提取器

    从SQL中提取日期函数调用。
    """

    # 所有支持的日期函数名
    DATE_FUNCTIONS = {
        'CURRENT_DATE', 'CURDATE', 'CURRENT_TIME', 'CURTIME', 'NOW', 'SYSDATE', 'GETDATE',
        'YEAR', 'MONTH', 'DAY', 'DAYOFWEEK', 'DAYOFMONTH', 'DAYOFYEAR',
        'WEEK', 'WEEKOFYEAR', 'QUARTER', 'HOUR', 'MINUTE', 'SECOND',
        'EXTRACT', 'DATE_FORMAT', 'TIME_FORMAT', 'TO_CHAR', 'FORMAT',
        'STR_TO_DATE', 'TO_DATE', 'PARSE_DATE',
        'DATE_ADD', 'DATE_SUB', 'ADDDATE', 'SUBDATE', 'DATEDIFF', 'TIMESTAMPDIFF', 'DATE_DIFF',
        'DATE_TRUNC', 'TRUNC', 'TRUNCATE',
        'MAKEDATE', 'MAKETIME', 'DATE', 'TIME',
        'LAST_DAY', 'FIRST_DAY', 'EOMONTH',
        'DATE_CMP', 'AGE', 'MONTHS_BETWEEN', 'ADD_MONTHS'
    }

    # 时间单位关键字
    TIME_UNITS = {
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
        'WEEK', 'QUARTER', 'MICROSECOND', 'MILLISECOND'
    }

    def extract(self, sql: str) -> List[DateFunctionCall]:
        """从SQL中提取日期函数调用

        Args:
            sql: SQL语句

        Returns:
            日期函数调用列表
        """
        functions = []

        for func_name in self.DATE_FUNCTIONS:
            # 匹配函数调用模式: FUNC_NAME(args) 或 FUNC_NAME (无参数)
            pattern = rf'\b{func_name}\s*(?:\(([^()]*(?:\([^()]*\)[^()]*)*)\))?'

            for match in re.finditer(pattern, sql, re.IGNORECASE):
                # 跳过没有括号的关键字（可能是列名或其他用途）
                if match.group(1) is None and '(' not in match.group(0):
                    # 但是某些函数可以不带括号，如CURRENT_DATE
                    if func_name not in ('CURRENT_DATE', 'CURRENT_TIME', 'NOW'):
                        continue

                func_call = DateFunctionCall(
                    func_type=DateFunctionType.from_string(func_name),
                    func_name=func_name,
                    raw_text=match.group(0)
                )

                # 解析参数
                args_str = match.group(1) or ""
                func_call.arguments = self._parse_arguments(args_str)

                # 提取列引用
                func_call.column_refs = self._extract_column_refs(args_str)

                # 提取时间单位
                func_call.time_unit = self._extract_time_unit(args_str)

                # 检查字面值
                literals = self._extract_literals(args_str)
                if literals:
                    func_call.has_literal = True
                    func_call.literal_values = literals

                functions.append(func_call)

        return functions

    def _parse_arguments(self, args_str: str) -> List[str]:
        """解析函数参数"""
        if not args_str.strip():
            return []

        # 简单分割（不处理嵌套函数）
        args = []
        depth = 0
        current = ""

        for char in args_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            args.append(current.strip())

        return args

    def _extract_column_refs(self, args_str: str) -> List[str]:
        """提取列引用"""
        # 匹配 table.column 或 column 格式
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'

        refs = []
        for match in re.finditer(pattern, args_str):
            ref = match.group(1)
            # 排除关键字、函数名和时间单位
            if ref.upper() not in self.DATE_FUNCTIONS and \
               ref.upper() not in self.TIME_UNITS and \
               ref.upper() not in ('AS', 'FROM', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE',
                                   'INTERVAL', 'DATE', 'TIME', 'TIMESTAMP'):
                refs.append(ref)

        return refs

    def _extract_time_unit(self, args_str: str) -> Optional[str]:
        """提取时间单位"""
        for unit in self.TIME_UNITS:
            if re.search(rf'\b{unit}\b', args_str, re.IGNORECASE):
                return unit
        return None

    def _extract_literals(self, args_str: str) -> List[str]:
        """提取字符串字面值"""
        # 匹配单引号或双引号包围的字符串
        pattern = r"'([^']*)'|\"([^\"]*)\""

        literals = []
        for match in re.finditer(pattern, args_str):
            literal = match.group(1) or match.group(2)
            literals.append(literal)

        return literals


class DateFunctionValidator:
    """日期函数校验器

    校验日期函数的安全性。
    """

    # 高精度时间提取函数（可能泄露精确时间信息）
    HIGH_PRECISION_FUNCTIONS = {
        DateFunctionType.HOUR,
        DateFunctionType.MINUTE,
        DateFunctionType.SECOND,
    }

    # 粗粒度时间提取函数（相对安全）
    COARSE_GRAIN_FUNCTIONS = {
        DateFunctionType.YEAR,
        DateFunctionType.QUARTER,
        DateFunctionType.MONTH,
    }

    # 日期差异函数（可能泄露时间间隔）
    DATE_DIFF_FUNCTIONS = {
        DateFunctionType.DATEDIFF,
        DateFunctionType.TIMESTAMPDIFF,
        DateFunctionType.DATE_DIFF,
        DateFunctionType.AGE,
        DateFunctionType.MONTHS_BETWEEN,
    }

    # 日期格式化函数（可能暴露原始日期）
    FORMAT_FUNCTIONS = {
        DateFunctionType.DATE_FORMAT,
        DateFunctionType.TIME_FORMAT,
        DateFunctionType.TO_CHAR,
        DateFunctionType.FORMAT,
    }

    def __init__(self, get_field_security_func=None):
        """初始化校验器

        Args:
            get_field_security_func: 获取字段安全配置的函数
        """
        self.get_field_security = get_field_security_func

    def validate(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """校验日期函数调用

        Args:
            func_call: 日期函数调用

        Returns:
            问题列表
        """
        issues = []

        # 检查高精度时间提取
        if func_call.func_type in self.HIGH_PRECISION_FUNCTIONS:
            issues.extend(self._check_high_precision_function(func_call))

        # 检查日期差异函数
        if func_call.func_type in self.DATE_DIFF_FUNCTIONS:
            issues.extend(self._check_date_diff_function(func_call))

        # 检查日期格式化函数
        if func_call.func_type in self.FORMAT_FUNCTIONS:
            issues.extend(self._check_format_function(func_call))

        # 检查EXTRACT函数的提取粒度
        if func_call.func_type == DateFunctionType.EXTRACT:
            issues.extend(self._check_extract_function(func_call))

        # 检查DATE_TRUNC函数
        if func_call.func_type in (DateFunctionType.DATE_TRUNC, DateFunctionType.TRUNC):
            issues.extend(self._check_trunc_function(func_call))

        # 检查涉及敏感字段的函数
        issues.extend(self._check_sensitive_columns(func_call))

        return issues

    def _check_high_precision_function(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查高精度时间提取函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.MEDIUM,
            description=f"高精度时间提取函数 {func_call.func_name}() 可能泄露精确时间信息",
            sql_fragment=func_call.raw_text,
            remediation="考虑使用粗粒度时间（如YEAR、MONTH）或对结果进行聚合",
            affected_fields=func_call.column_refs,
            validator_name="DateFunctionValidator"
        ))

        return issues

    def _check_date_diff_function(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查日期差异函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.LOW,
            description=f"日期差异函数 {func_call.func_name}() 可能间接泄露原始日期信息",
            sql_fragment=func_call.raw_text,
            remediation="确保日期差异结果不会被用于推断具体日期",
            affected_fields=func_call.column_refs,
            validator_name="DateFunctionValidator"
        ))

        return issues

    def _check_format_function(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查日期格式化函数"""
        issues = []

        # 检查格式化精度
        high_precision_formats = ['%H', '%i', '%s', '%S', 'HH', 'MI', 'SS']
        has_high_precision = any(
            fmt in literal for literal in func_call.literal_values
            for fmt in high_precision_formats
        )

        if has_high_precision:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
                risk_level=RiskLevel.MEDIUM,
                description=f"日期格式化函数 {func_call.func_name}() 使用了高精度时间格式",
                sql_fragment=func_call.raw_text,
                remediation="避免在联邦查询中暴露精确时间（时、分、秒）",
                affected_fields=func_call.column_refs,
                validator_name="DateFunctionValidator"
            ))
        else:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"日期格式化函数 {func_call.func_name}() 可能暴露日期信息",
                sql_fragment=func_call.raw_text,
                remediation="确保格式化后的日期不会泄露隐私",
                affected_fields=func_call.column_refs,
                validator_name="DateFunctionValidator"
            ))

        return issues

    def _check_extract_function(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查EXTRACT函数"""
        issues = []

        high_precision_units = {'HOUR', 'MINUTE', 'SECOND', 'MILLISECOND', 'MICROSECOND'}

        if func_call.time_unit and func_call.time_unit.upper() in high_precision_units:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.MEDIUM,
                description=f"EXTRACT函数提取高精度时间单位 {func_call.time_unit}",
                sql_fragment=func_call.raw_text,
                remediation="考虑提取粗粒度时间单位（YEAR、MONTH、DAY）",
                affected_fields=func_call.column_refs,
                validator_name="DateFunctionValidator"
            ))

        return issues

    def _check_trunc_function(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查DATE_TRUNC函数"""
        issues = []

        # DATE_TRUNC通常是好的做法，但需要检查截断粒度
        fine_grain_units = {'HOUR', 'MINUTE', 'SECOND'}

        if func_call.time_unit and func_call.time_unit.upper() in fine_grain_units:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"DATE_TRUNC使用细粒度截断 {func_call.time_unit}",
                sql_fragment=func_call.raw_text,
                remediation="考虑使用更粗粒度的截断（DAY、WEEK、MONTH）",
                affected_fields=func_call.column_refs,
                validator_name="DateFunctionValidator"
            ))

        return issues

    def _check_sensitive_columns(self, func_call: DateFunctionCall) -> List[ValidationIssue]:
        """检查涉及敏感字段的函数"""
        issues = []

        if not self.get_field_security:
            return issues

        for col_ref in func_call.column_refs:
            parts = col_ref.split('.')
            table_name = parts[0] if len(parts) > 1 else ""
            field_name = parts[-1]

            try:
                field_meta = self.get_field_security(table_name, field_name)
                if field_meta and field_meta.security_level in (
                    FieldSecurityLevel.PRIVATE, FieldSecurityLevel.SENSITIVE
                ):
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.DIRECT_FIELD_EXPOSURE,
                        risk_level=RiskLevel.HIGH,
                        description=f"日期函数 {func_call.func_name}() 操作敏感日期字段 {col_ref}",
                        sql_fragment=func_call.raw_text,
                        remediation="敏感日期字段应使用DATE_TRUNC进行泛化处理",
                        affected_fields=[col_ref],
                        validator_name="DateFunctionValidator"
                    ))
            except Exception:
                pass

        return issues


# 日期函数安全规则
DATE_FUNCTION_RULES = {
    DateFunctionType.YEAR: {
        "risk_level": RiskLevel.SAFE,
        "description": "提取年份，粗粒度时间，通常安全"
    },
    DateFunctionType.QUARTER: {
        "risk_level": RiskLevel.SAFE,
        "description": "提取季度，粗粒度时间，通常安全"
    },
    DateFunctionType.MONTH: {
        "risk_level": RiskLevel.LOW,
        "description": "提取月份，相对安全"
    },
    DateFunctionType.DAY: {
        "risk_level": RiskLevel.LOW,
        "description": "提取日期，中等粒度"
    },
    DateFunctionType.DAYOFWEEK: {
        "risk_level": RiskLevel.SAFE,
        "description": "提取星期几，通常安全"
    },
    DateFunctionType.HOUR: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "提取小时，细粒度时间，需谨慎"
    },
    DateFunctionType.MINUTE: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "提取分钟，高精度时间"
    },
    DateFunctionType.SECOND: {
        "risk_level": RiskLevel.HIGH,
        "description": "提取秒数，高精度时间，风险较高"
    },
    DateFunctionType.DATEDIFF: {
        "risk_level": RiskLevel.LOW,
        "description": "计算日期差异，可能间接泄露日期"
    },
    DateFunctionType.DATE_TRUNC: {
        "risk_level": RiskLevel.SAFE,
        "description": "日期截断，隐私保护的好方法"
    },
    DateFunctionType.DATE_FORMAT: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "日期格式化，取决于格式精度"
    },
    DateFunctionType.NOW: {
        "risk_level": RiskLevel.SAFE,
        "description": "获取当前时间，通常安全"
    },
    DateFunctionType.CURRENT_DATE: {
        "risk_level": RiskLevel.SAFE,
        "description": "获取当前日期，通常安全"
    },
}


def get_date_function_risk(func_type: DateFunctionType) -> RiskLevel:
    """获取日期函数的风险等级"""
    rule = DATE_FUNCTION_RULES.get(func_type)
    if rule:
        return rule["risk_level"]
    return RiskLevel.LOW


def is_safe_date_function(func_name: str) -> bool:
    """检查日期函数是否安全"""
    func_type = DateFunctionType.from_string(func_name)
    return get_date_function_risk(func_type) == RiskLevel.SAFE
