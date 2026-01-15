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
时间戳类型函数模块

提供联邦分析场景下时间戳函数的安全校验和处理。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..field_security import FieldMeta, FieldSecurityLevel


class TimestampFunctionType(Enum):
    """时间戳函数类型"""
    # 当前时间戳
    CURRENT_TIMESTAMP = "CURRENT_TIMESTAMP"
    LOCALTIMESTAMP = "LOCALTIMESTAMP"
    SYSTIMESTAMP = "SYSTIMESTAMP"
    GETUTCDATE = "GETUTCDATE"

    # 时间戳创建
    TIMESTAMP = "TIMESTAMP"
    MAKE_TIMESTAMP = "MAKE_TIMESTAMP"
    TO_TIMESTAMP = "TO_TIMESTAMP"
    FROM_UNIXTIME = "FROM_UNIXTIME"
    TIMESTAMP_FROM_UNIX = "TIMESTAMP_FROM_UNIX"

    # 时间戳转换
    UNIX_TIMESTAMP = "UNIX_TIMESTAMP"
    TO_UNIX_TIMESTAMP = "TO_UNIX_TIMESTAMP"
    EXTRACT_EPOCH = "EXTRACT_EPOCH"

    # 时间戳运算
    TIMESTAMP_ADD = "TIMESTAMP_ADD"
    TIMESTAMP_SUB = "TIMESTAMP_SUB"
    TIMESTAMPADD = "TIMESTAMPADD"
    TIMESTAMPSUB = "TIMESTAMPSUB"

    # 时间戳差异
    TIMESTAMP_DIFF = "TIMESTAMP_DIFF"
    TIMESTAMPDIFF = "TIMESTAMPDIFF"

    # 时间戳截断
    TIMESTAMP_TRUNC = "TIMESTAMP_TRUNC"

    # 时间戳格式化
    TIMESTAMP_FORMAT = "TIMESTAMP_FORMAT"
    TO_CHAR = "TO_CHAR"

    # 时区相关
    AT_TIME_ZONE = "AT_TIME_ZONE"
    CONVERT_TZ = "CONVERT_TZ"
    CONVERT_TIMEZONE = "CONVERT_TIMEZONE"
    TIMEZONE = "TIMEZONE"

    # 时间戳比较
    TIMESTAMP_CMP = "TIMESTAMP_CMP"

    # 毫秒/微秒提取
    EXTRACT_MILLISECOND = "EXTRACT_MILLISECOND"
    EXTRACT_MICROSECOND = "EXTRACT_MICROSECOND"

    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, func_name: str) -> 'TimestampFunctionType':
        """从函数名转换"""
        name_upper = func_name.upper()
        for member in cls:
            if member.value == name_upper:
                return member
        return cls.UNKNOWN


@dataclass
class TimestampFunctionCall:
    """时间戳函数调用"""
    func_type: TimestampFunctionType
    func_name: str
    arguments: List[Any] = field(default_factory=list)
    raw_text: str = ""

    # 涉及的列
    column_refs: List[str] = field(default_factory=list)

    # 时间单位（用于TIMESTAMP_DIFF等）
    time_unit: Optional[str] = None

    # 时区信息
    timezone: Optional[str] = None

    # 是否包含字面值
    has_literal: bool = False
    literal_values: List[str] = field(default_factory=list)


class TimestampFunctionExtractor:
    """时间戳函数提取器

    从SQL中提取时间戳函数调用。
    """

    # 所有支持的时间戳函数名
    TIMESTAMP_FUNCTIONS = {
        'CURRENT_TIMESTAMP', 'LOCALTIMESTAMP', 'SYSTIMESTAMP', 'GETUTCDATE',
        'TIMESTAMP', 'MAKE_TIMESTAMP', 'TO_TIMESTAMP', 'FROM_UNIXTIME', 'TIMESTAMP_FROM_UNIX',
        'UNIX_TIMESTAMP', 'TO_UNIX_TIMESTAMP', 'EXTRACT_EPOCH',
        'TIMESTAMP_ADD', 'TIMESTAMP_SUB', 'TIMESTAMPADD', 'TIMESTAMPSUB',
        'TIMESTAMP_DIFF', 'TIMESTAMPDIFF',
        'TIMESTAMP_TRUNC',
        'TIMESTAMP_FORMAT', 'TO_CHAR',
        'AT_TIME_ZONE', 'CONVERT_TZ', 'CONVERT_TIMEZONE', 'TIMEZONE',
        'TIMESTAMP_CMP'
    }

    # 时间单位关键字
    TIME_UNITS = {
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
        'WEEK', 'QUARTER', 'MICROSECOND', 'MILLISECOND', 'NANOSECOND'
    }

    # 常见时区
    TIMEZONES = {
        'UTC', 'GMT', 'EST', 'PST', 'CST', 'MST',
        'America/New_York', 'America/Los_Angeles', 'Europe/London',
        'Asia/Shanghai', 'Asia/Tokyo'
    }

    def extract(self, sql: str) -> List[TimestampFunctionCall]:
        """从SQL中提取时间戳函数调用

        Args:
            sql: SQL语句

        Returns:
            时间戳函数调用列表
        """
        functions = []

        for func_name in self.TIMESTAMP_FUNCTIONS:
            # 匹配函数调用模式
            pattern = rf'\b{func_name}\s*(?:\(([^()]*(?:\([^()]*\)[^()]*)*)\))?'

            for match in re.finditer(pattern, sql, re.IGNORECASE):
                # 跳过没有括号的关键字
                if match.group(1) is None and '(' not in match.group(0):
                    if func_name not in ('CURRENT_TIMESTAMP', 'LOCALTIMESTAMP'):
                        continue

                func_call = TimestampFunctionCall(
                    func_type=TimestampFunctionType.from_string(func_name),
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

                # 提取时区
                func_call.timezone = self._extract_timezone(args_str)

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
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'

        refs = []
        for match in re.finditer(pattern, args_str):
            ref = match.group(1)
            if ref.upper() not in self.TIMESTAMP_FUNCTIONS and \
               ref.upper() not in self.TIME_UNITS and \
               ref.upper() not in ('AS', 'FROM', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE',
                                   'INTERVAL', 'DATE', 'TIME', 'TIMESTAMP', 'AT', 'ZONE'):
                refs.append(ref)

        return refs

    def _extract_time_unit(self, args_str: str) -> Optional[str]:
        """提取时间单位"""
        for unit in self.TIME_UNITS:
            if re.search(rf'\b{unit}\b', args_str, re.IGNORECASE):
                return unit
        return None

    def _extract_timezone(self, args_str: str) -> Optional[str]:
        """提取时区"""
        # 匹配常见时区格式
        tz_pattern = r"'([A-Za-z_/]+)'|\"([A-Za-z_/]+)\""
        for match in re.finditer(tz_pattern, args_str):
            tz = match.group(1) or match.group(2)
            if tz.upper() in self.TIMEZONES or '/' in tz:
                return tz
        return None

    def _extract_literals(self, args_str: str) -> List[str]:
        """提取字符串字面值"""
        pattern = r"'([^']*)'|\"([^\"]*)\""

        literals = []
        for match in re.finditer(pattern, args_str):
            literal = match.group(1) or match.group(2)
            literals.append(literal)

        return literals


class TimestampFunctionValidator:
    """时间戳函数校验器

    校验时间戳函数的安全性。
    """

    # Unix时间戳函数（可能泄露精确时间）
    UNIX_TIMESTAMP_FUNCTIONS = {
        TimestampFunctionType.UNIX_TIMESTAMP,
        TimestampFunctionType.TO_UNIX_TIMESTAMP,
        TimestampFunctionType.EXTRACT_EPOCH,
        TimestampFunctionType.FROM_UNIXTIME,
    }

    # 时间戳差异函数
    TIMESTAMP_DIFF_FUNCTIONS = {
        TimestampFunctionType.TIMESTAMP_DIFF,
        TimestampFunctionType.TIMESTAMPDIFF,
    }

    # 时区转换函数（可能暴露地理位置）
    TIMEZONE_FUNCTIONS = {
        TimestampFunctionType.AT_TIME_ZONE,
        TimestampFunctionType.CONVERT_TZ,
        TimestampFunctionType.CONVERT_TIMEZONE,
        TimestampFunctionType.TIMEZONE,
    }

    # 高精度时间戳函数
    HIGH_PRECISION_FUNCTIONS = {
        TimestampFunctionType.EXTRACT_MILLISECOND,
        TimestampFunctionType.EXTRACT_MICROSECOND,
    }

    def __init__(self, get_field_security_func=None):
        """初始化校验器

        Args:
            get_field_security_func: 获取字段安全配置的函数
        """
        self.get_field_security = get_field_security_func

    def validate(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """校验时间戳函数调用

        Args:
            func_call: 时间戳函数调用

        Returns:
            问题列表
        """
        issues = []

        # 检查Unix时间戳函数
        if func_call.func_type in self.UNIX_TIMESTAMP_FUNCTIONS:
            issues.extend(self._check_unix_timestamp_function(func_call))

        # 检查时间戳差异函数
        if func_call.func_type in self.TIMESTAMP_DIFF_FUNCTIONS:
            issues.extend(self._check_timestamp_diff_function(func_call))

        # 检查时区函数
        if func_call.func_type in self.TIMEZONE_FUNCTIONS:
            issues.extend(self._check_timezone_function(func_call))

        # 检查高精度函数
        if func_call.func_type in self.HIGH_PRECISION_FUNCTIONS:
            issues.extend(self._check_high_precision_function(func_call))

        # 检查时间戳截断
        if func_call.func_type == TimestampFunctionType.TIMESTAMP_TRUNC:
            issues.extend(self._check_trunc_function(func_call))

        # 检查涉及敏感字段的函数
        issues.extend(self._check_sensitive_columns(func_call))

        return issues

    def _check_unix_timestamp_function(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """检查Unix时间戳函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.HIGH,
            description=f"Unix时间戳函数 {func_call.func_name}() 会暴露精确到秒的时间信息",
            sql_fragment=func_call.raw_text,
            remediation="避免在联邦查询中直接返回Unix时间戳，考虑使用TIMESTAMP_TRUNC进行泛化",
            affected_fields=func_call.column_refs,
            validator_name="TimestampFunctionValidator"
        ))

        return issues

    def _check_timestamp_diff_function(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """检查时间戳差异函数"""
        issues = []

        # 检查差异精度
        high_precision_units = {'SECOND', 'MILLISECOND', 'MICROSECOND', 'NANOSECOND'}

        if func_call.time_unit and func_call.time_unit.upper() in high_precision_units:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.MEDIUM,
                description=f"时间戳差异函数使用高精度单位 {func_call.time_unit}",
                sql_fragment=func_call.raw_text,
                remediation="考虑使用更粗粒度的时间单位（DAY、HOUR）",
                affected_fields=func_call.column_refs,
                validator_name="TimestampFunctionValidator"
            ))
        else:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"时间戳差异函数 {func_call.func_name}() 可能间接泄露时间信息",
                sql_fragment=func_call.raw_text,
                remediation="确保时间差异结果不会被用于推断具体时间",
                affected_fields=func_call.column_refs,
                validator_name="TimestampFunctionValidator"
            ))

        return issues

    def _check_timezone_function(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """检查时区转换函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
            risk_level=RiskLevel.MEDIUM,
            description=f"时区转换函数 {func_call.func_name}() 可能暴露用户地理位置信息",
            sql_fragment=func_call.raw_text,
            remediation="时区信息可能泄露用户位置，考虑统一使用UTC时间",
            affected_fields=func_call.column_refs,
            validator_name="TimestampFunctionValidator"
        ))

        return issues

    def _check_high_precision_function(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """检查高精度时间戳函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.HIGH,
            description=f"高精度时间戳函数 {func_call.func_name}() 可能泄露精确时间",
            sql_fragment=func_call.raw_text,
            remediation="毫秒/微秒级时间精度可能被用于唯一标识记录",
            affected_fields=func_call.column_refs,
            validator_name="TimestampFunctionValidator"
        ))

        return issues

    def _check_trunc_function(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
        """检查时间戳截断函数"""
        issues = []

        # TIMESTAMP_TRUNC通常是好的做法
        fine_grain_units = {'MINUTE', 'SECOND', 'MILLISECOND', 'MICROSECOND'}

        if func_call.time_unit and func_call.time_unit.upper() in fine_grain_units:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"TIMESTAMP_TRUNC使用细粒度截断 {func_call.time_unit}",
                sql_fragment=func_call.raw_text,
                remediation="建议使用更粗粒度的截断（HOUR、DAY）以增强隐私保护",
                affected_fields=func_call.column_refs,
                validator_name="TimestampFunctionValidator"
            ))

        return issues

    def _check_sensitive_columns(self, func_call: TimestampFunctionCall) -> List[ValidationIssue]:
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
                        description=f"时间戳函数 {func_call.func_name}() 操作敏感时间戳字段 {col_ref}",
                        sql_fragment=func_call.raw_text,
                        remediation="敏感时间戳字段应使用TIMESTAMP_TRUNC进行粗粒度处理",
                        affected_fields=[col_ref],
                        validator_name="TimestampFunctionValidator"
                    ))
            except Exception:
                pass

        return issues


# 时间戳函数安全规则
TIMESTAMP_FUNCTION_RULES = {
    TimestampFunctionType.CURRENT_TIMESTAMP: {
        "risk_level": RiskLevel.SAFE,
        "description": "获取当前时间戳，通常安全"
    },
    TimestampFunctionType.LOCALTIMESTAMP: {
        "risk_level": RiskLevel.LOW,
        "description": "获取本地时间戳，可能暴露时区"
    },
    TimestampFunctionType.UNIX_TIMESTAMP: {
        "risk_level": RiskLevel.HIGH,
        "description": "转换为Unix时间戳，精确到秒"
    },
    TimestampFunctionType.FROM_UNIXTIME: {
        "risk_level": RiskLevel.HIGH,
        "description": "从Unix时间戳转换，暴露精确时间"
    },
    TimestampFunctionType.TIMESTAMP_DIFF: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "计算时间戳差异，取决于精度"
    },
    TimestampFunctionType.TIMESTAMP_TRUNC: {
        "risk_level": RiskLevel.SAFE,
        "description": "时间戳截断，隐私保护的好方法"
    },
    TimestampFunctionType.CONVERT_TZ: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "时区转换，可能暴露地理位置"
    },
    TimestampFunctionType.TO_TIMESTAMP: {
        "risk_level": RiskLevel.LOW,
        "description": "转换为时间戳类型"
    },
}


def get_timestamp_function_risk(func_type: TimestampFunctionType) -> RiskLevel:
    """获取时间戳函数的风险等级"""
    rule = TIMESTAMP_FUNCTION_RULES.get(func_type)
    if rule:
        return rule["risk_level"]
    return RiskLevel.LOW


def is_safe_timestamp_function(func_name: str) -> bool:
    """检查时间戳函数是否安全"""
    func_type = TimestampFunctionType.from_string(func_name)
    return get_timestamp_function_risk(func_type) == RiskLevel.SAFE
