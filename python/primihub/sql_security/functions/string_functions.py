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
字符类型函数模块

提供联邦分析场景下字符串函数的安全校验和处理。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..field_security import FieldMeta, FieldSecurityLevel


class StringFunctionType(Enum):
    """字符串函数类型"""
    # 字符串操作
    CONCAT = "CONCAT"
    CONCAT_WS = "CONCAT_WS"
    SUBSTRING = "SUBSTRING"
    SUBSTR = "SUBSTR"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    # 大小写转换
    UPPER = "UPPER"
    LOWER = "LOWER"
    INITCAP = "INITCAP"

    # 空白处理
    TRIM = "TRIM"
    LTRIM = "LTRIM"
    RTRIM = "RTRIM"

    # 填充
    LPAD = "LPAD"
    RPAD = "RPAD"

    # 查找替换
    REPLACE = "REPLACE"
    TRANSLATE = "TRANSLATE"
    INSTR = "INSTR"
    LOCATE = "LOCATE"
    POSITION = "POSITION"

    # 长度和信息
    LENGTH = "LENGTH"
    CHAR_LENGTH = "CHAR_LENGTH"
    CHARACTER_LENGTH = "CHARACTER_LENGTH"
    OCTET_LENGTH = "OCTET_LENGTH"
    BIT_LENGTH = "BIT_LENGTH"

    # 格式化
    FORMAT = "FORMAT"
    PRINTF = "PRINTF"

    # 编码
    ASCII = "ASCII"
    CHR = "CHR"
    CHAR = "CHAR"

    # 反转和重复
    REVERSE = "REVERSE"
    REPEAT = "REPEAT"
    SPACE = "SPACE"

    # 正则表达式
    REGEXP_REPLACE = "REGEXP_REPLACE"
    REGEXP_SUBSTR = "REGEXP_SUBSTR"
    REGEXP_INSTR = "REGEXP_INSTR"
    REGEXP_LIKE = "REGEXP_LIKE"
    REGEXP_MATCH = "REGEXP_MATCH"

    # 其他
    COALESCE = "COALESCE"
    NULLIF = "NULLIF"
    NVL = "NVL"
    IFNULL = "IFNULL"

    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, func_name: str) -> 'StringFunctionType':
        """从函数名转换"""
        name_upper = func_name.upper()
        for member in cls:
            if member.value == name_upper:
                return member
        return cls.UNKNOWN


@dataclass
class StringFunctionCall:
    """字符串函数调用"""
    func_type: StringFunctionType
    func_name: str
    arguments: List[Any] = field(default_factory=list)
    raw_text: str = ""

    # 涉及的列
    column_refs: List[str] = field(default_factory=list)

    # 是否包含字面值
    has_literal: bool = False
    literal_values: List[str] = field(default_factory=list)


class StringFunctionExtractor:
    """字符串函数提取器

    从SQL中提取字符串函数调用。
    """

    # 所有支持的字符串函数名
    STRING_FUNCTIONS = {
        'CONCAT', 'CONCAT_WS', 'SUBSTRING', 'SUBSTR', 'LEFT', 'RIGHT',
        'UPPER', 'LOWER', 'INITCAP', 'TRIM', 'LTRIM', 'RTRIM',
        'LPAD', 'RPAD', 'REPLACE', 'TRANSLATE', 'INSTR', 'LOCATE', 'POSITION',
        'LENGTH', 'CHAR_LENGTH', 'CHARACTER_LENGTH', 'OCTET_LENGTH', 'BIT_LENGTH',
        'FORMAT', 'PRINTF', 'ASCII', 'CHR', 'CHAR', 'REVERSE', 'REPEAT', 'SPACE',
        'REGEXP_REPLACE', 'REGEXP_SUBSTR', 'REGEXP_INSTR', 'REGEXP_LIKE', 'REGEXP_MATCH',
        'COALESCE', 'NULLIF', 'NVL', 'IFNULL'
    }

    def extract(self, sql: str) -> List[StringFunctionCall]:
        """从SQL中提取字符串函数调用

        Args:
            sql: SQL语句

        Returns:
            字符串函数调用列表
        """
        functions = []

        for func_name in self.STRING_FUNCTIONS:
            # 匹配函数调用模式: FUNC_NAME(args)
            pattern = rf'\b{func_name}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)'

            for match in re.finditer(pattern, sql, re.IGNORECASE):
                func_call = StringFunctionCall(
                    func_type=StringFunctionType.from_string(func_name),
                    func_name=func_name,
                    raw_text=match.group(0)
                )

                # 解析参数
                args_str = match.group(1)
                func_call.arguments = self._parse_arguments(args_str)

                # 提取列引用
                func_call.column_refs = self._extract_column_refs(args_str)

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
            # 排除关键字和函数名
            if ref.upper() not in self.STRING_FUNCTIONS and \
               ref.upper() not in ('AS', 'FROM', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE'):
                refs.append(ref)

        return refs

    def _extract_literals(self, args_str: str) -> List[str]:
        """提取字符串字面值"""
        # 匹配单引号或双引号包围的字符串
        pattern = r"'([^']*)'|\"([^\"]*)\""

        literals = []
        for match in re.finditer(pattern, args_str):
            literal = match.group(1) or match.group(2)
            literals.append(literal)

        return literals


class StringFunctionValidator:
    """字符串函数校验器

    校验字符串函数的安全性。
    """

    # 高风险函数（可能泄露信息）
    HIGH_RISK_FUNCTIONS = {
        StringFunctionType.SUBSTRING,
        StringFunctionType.SUBSTR,
        StringFunctionType.LEFT,
        StringFunctionType.RIGHT,
        StringFunctionType.REGEXP_SUBSTR,
        StringFunctionType.REGEXP_REPLACE,
    }

    # 信息泄露函数
    INFO_LEAK_FUNCTIONS = {
        StringFunctionType.LENGTH,
        StringFunctionType.CHAR_LENGTH,
        StringFunctionType.CHARACTER_LENGTH,
        StringFunctionType.INSTR,
        StringFunctionType.LOCATE,
        StringFunctionType.POSITION,
    }

    def __init__(self, get_field_security_func=None):
        """初始化校验器

        Args:
            get_field_security_func: 获取字段安全配置的函数
        """
        self.get_field_security = get_field_security_func

    def validate(self, func_call: StringFunctionCall) -> List[ValidationIssue]:
        """校验字符串函数调用

        Args:
            func_call: 字符串函数调用

        Returns:
            问题列表
        """
        issues = []

        # 检查高风险函数
        if func_call.func_type in self.HIGH_RISK_FUNCTIONS:
            issues.extend(self._check_high_risk_function(func_call))

        # 检查信息泄露函数
        if func_call.func_type in self.INFO_LEAK_FUNCTIONS:
            issues.extend(self._check_info_leak_function(func_call))

        # 检查正则表达式函数
        if func_call.func_type in (
            StringFunctionType.REGEXP_REPLACE,
            StringFunctionType.REGEXP_SUBSTR,
            StringFunctionType.REGEXP_LIKE,
            StringFunctionType.REGEXP_MATCH
        ):
            issues.extend(self._check_regex_function(func_call))

        # 检查涉及敏感字段的函数
        issues.extend(self._check_sensitive_columns(func_call))

        return issues

    def _check_high_risk_function(self, func_call: StringFunctionCall) -> List[ValidationIssue]:
        """检查高风险函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
            risk_level=RiskLevel.MEDIUM,
            description=f"字符串截取函数 {func_call.func_name}() 可能泄露部分敏感信息",
            sql_fragment=func_call.raw_text,
            remediation="避免对敏感字段使用字符串截取操作",
            affected_fields=func_call.column_refs,
            validator_name="StringFunctionValidator"
        ))

        return issues

    def _check_info_leak_function(self, func_call: StringFunctionCall) -> List[ValidationIssue]:
        """检查信息泄露函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.LOW,
            description=f"函数 {func_call.func_name}() 可能泄露字段长度或位置信息",
            sql_fragment=func_call.raw_text,
            remediation="长度和位置信息可能被用于推断原始值",
            affected_fields=func_call.column_refs,
            validator_name="StringFunctionValidator"
        ))

        return issues

    def _check_regex_function(self, func_call: StringFunctionCall) -> List[ValidationIssue]:
        """检查正则表达式函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.SENSITIVE_FILTER_CONDITION,
            risk_level=RiskLevel.MEDIUM,
            description=f"正则表达式函数 {func_call.func_name}() 可能用于探测敏感数据模式",
            sql_fragment=func_call.raw_text,
            remediation="正则表达式匹配可能被滥用来探测数据格式和内容",
            affected_fields=func_call.column_refs,
            validator_name="StringFunctionValidator"
        ))

        return issues

    def _check_sensitive_columns(self, func_call: StringFunctionCall) -> List[ValidationIssue]:
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
                if field_meta and field_meta.requires_encryption():
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.DIRECT_FIELD_EXPOSURE,
                        risk_level=RiskLevel.HIGH,
                        description=f"字符串函数 {func_call.func_name}() 操作敏感字段 {col_ref}",
                        sql_fragment=func_call.raw_text,
                        remediation="敏感字段不应直接进行字符串操作",
                        affected_fields=[col_ref],
                        validator_name="StringFunctionValidator"
                    ))
            except Exception:
                pass

        return issues


# 字符串函数安全规则
STRING_FUNCTION_RULES = {
    StringFunctionType.CONCAT: {
        "risk_level": RiskLevel.LOW,
        "description": "字符串连接函数，可能用于组合敏感信息"
    },
    StringFunctionType.SUBSTRING: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "字符串截取函数，可能泄露部分敏感数据"
    },
    StringFunctionType.LENGTH: {
        "risk_level": RiskLevel.LOW,
        "description": "获取字符串长度，可能泄露数据规模信息"
    },
    StringFunctionType.UPPER: {
        "risk_level": RiskLevel.SAFE,
        "description": "大写转换，通常安全"
    },
    StringFunctionType.LOWER: {
        "risk_level": RiskLevel.SAFE,
        "description": "小写转换，通常安全"
    },
    StringFunctionType.TRIM: {
        "risk_level": RiskLevel.SAFE,
        "description": "去除空白，通常安全"
    },
    StringFunctionType.REPLACE: {
        "risk_level": RiskLevel.LOW,
        "description": "字符串替换，可能用于数据脱敏"
    },
    StringFunctionType.REGEXP_LIKE: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "正则匹配，可能用于探测数据模式"
    },
}


def get_string_function_risk(func_type: StringFunctionType) -> RiskLevel:
    """获取字符串函数的风险等级"""
    rule = STRING_FUNCTION_RULES.get(func_type)
    if rule:
        return rule["risk_level"]
    return RiskLevel.LOW


def is_safe_string_function(func_name: str) -> bool:
    """检查字符串函数是否安全"""
    func_type = StringFunctionType.from_string(func_name)
    return get_string_function_risk(func_type) == RiskLevel.SAFE
