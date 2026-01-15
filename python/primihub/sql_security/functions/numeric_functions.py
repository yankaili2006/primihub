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
浮点/数值类型函数模块

提供联邦分析场景下数值函数的安全校验和处理。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..validation_result import ValidationIssue, ValidationIssueType, RiskLevel
from ..field_security import FieldMeta, FieldSecurityLevel


class NumericFunctionType(Enum):
    """数值函数类型"""
    # 数学运算
    ABS = "ABS"
    CEIL = "CEIL"
    CEILING = "CEILING"
    FLOOR = "FLOOR"
    ROUND = "ROUND"
    TRUNCATE = "TRUNCATE"
    TRUNC = "TRUNC"

    # 幂和对数
    POWER = "POWER"
    POW = "POW"
    SQRT = "SQRT"
    EXP = "EXP"
    LOG = "LOG"
    LOG10 = "LOG10"
    LOG2 = "LOG2"
    LN = "LN"

    # 三角函数
    SIN = "SIN"
    COS = "COS"
    TAN = "TAN"
    ASIN = "ASIN"
    ACOS = "ACOS"
    ATAN = "ATAN"
    ATAN2 = "ATAN2"
    COT = "COT"

    # 符号和模
    SIGN = "SIGN"
    MOD = "MOD"

    # 随机数
    RAND = "RAND"
    RANDOM = "RANDOM"

    # 类型转换
    CAST = "CAST"
    CONVERT = "CONVERT"
    TO_NUMBER = "TO_NUMBER"
    TO_DECIMAL = "TO_DECIMAL"
    TO_FLOAT = "TO_FLOAT"
    TO_DOUBLE = "TO_DOUBLE"

    # 精度控制
    DECIMAL = "DECIMAL"
    NUMERIC = "NUMERIC"

    # 范围函数
    GREATEST = "GREATEST"
    LEAST = "LEAST"

    # 条件函数
    NULLIF = "NULLIF"
    COALESCE = "COALESCE"
    NVL = "NVL"
    IFNULL = "IFNULL"
    IF = "IF"
    CASE = "CASE"

    # 位运算
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BIT_NOT = "BIT_NOT"
    BITWISE_AND = "BITWISE_AND"
    BITWISE_OR = "BITWISE_OR"
    BITWISE_XOR = "BITWISE_XOR"

    # 其他
    PI = "PI"
    RADIANS = "RADIANS"
    DEGREES = "DEGREES"

    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, func_name: str) -> 'NumericFunctionType':
        """从函数名转换"""
        name_upper = func_name.upper()
        for member in cls:
            if member.value == name_upper:
                return member
        return cls.UNKNOWN


@dataclass
class NumericFunctionCall:
    """数值函数调用"""
    func_type: NumericFunctionType
    func_name: str
    arguments: List[Any] = field(default_factory=list)
    raw_text: str = ""

    # 涉及的列
    column_refs: List[str] = field(default_factory=list)

    # 精度参数（用于ROUND, TRUNCATE等）
    precision: Optional[int] = None

    # 目标类型（用于CAST, CONVERT）
    target_type: Optional[str] = None

    # 是否包含字面值
    has_literal: bool = False
    literal_values: List[str] = field(default_factory=list)


class NumericFunctionExtractor:
    """数值函数提取器

    从SQL中提取数值函数调用。
    """

    # 所有支持的数值函数名
    NUMERIC_FUNCTIONS = {
        'ABS', 'CEIL', 'CEILING', 'FLOOR', 'ROUND', 'TRUNCATE', 'TRUNC',
        'POWER', 'POW', 'SQRT', 'EXP', 'LOG', 'LOG10', 'LOG2', 'LN',
        'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN', 'ATAN2', 'COT',
        'SIGN', 'MOD',
        'RAND', 'RANDOM',
        'CAST', 'CONVERT', 'TO_NUMBER', 'TO_DECIMAL', 'TO_FLOAT', 'TO_DOUBLE',
        'GREATEST', 'LEAST',
        'NULLIF', 'COALESCE', 'NVL', 'IFNULL',
        'BIT_AND', 'BIT_OR', 'BIT_XOR', 'BIT_NOT',
        'PI', 'RADIANS', 'DEGREES'
    }

    # 数据类型关键字
    DATA_TYPES = {
        'INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT',
        'FLOAT', 'DOUBLE', 'REAL', 'DECIMAL', 'NUMERIC',
        'NUMBER', 'MONEY', 'SMALLMONEY'
    }

    def extract(self, sql: str) -> List[NumericFunctionCall]:
        """从SQL中提取数值函数调用

        Args:
            sql: SQL语句

        Returns:
            数值函数调用列表
        """
        functions = []

        for func_name in self.NUMERIC_FUNCTIONS:
            # 匹配函数调用模式
            pattern = rf'\b{func_name}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)'

            for match in re.finditer(pattern, sql, re.IGNORECASE):
                func_call = NumericFunctionCall(
                    func_type=NumericFunctionType.from_string(func_name),
                    func_name=func_name,
                    raw_text=match.group(0)
                )

                # 解析参数
                args_str = match.group(1)
                func_call.arguments = self._parse_arguments(args_str)

                # 提取列引用
                func_call.column_refs = self._extract_column_refs(args_str)

                # 提取精度参数
                func_call.precision = self._extract_precision(func_call)

                # 提取目标类型
                func_call.target_type = self._extract_target_type(args_str)

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
            if ref.upper() not in self.NUMERIC_FUNCTIONS and \
               ref.upper() not in self.DATA_TYPES and \
               ref.upper() not in ('AS', 'FROM', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE',
                                   'SIGNED', 'UNSIGNED', 'PRECISION'):
                refs.append(ref)

        return refs

    def _extract_precision(self, func_call: NumericFunctionCall) -> Optional[int]:
        """提取精度参数"""
        if func_call.func_type in (NumericFunctionType.ROUND, NumericFunctionType.TRUNCATE,
                                   NumericFunctionType.TRUNC):
            if len(func_call.arguments) >= 2:
                try:
                    return int(func_call.arguments[1])
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_target_type(self, args_str: str) -> Optional[str]:
        """提取目标类型（用于CAST）"""
        # 匹配 AS TYPE 模式
        as_pattern = r'\bAS\s+(\w+(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)'
        match = re.search(as_pattern, args_str, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_literals(self, args_str: str) -> List[str]:
        """提取数值字面值"""
        # 匹配数字（整数或浮点数）
        pattern = r'\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b'

        literals = []
        for match in re.finditer(pattern, args_str):
            literals.append(match.group(1))

        return literals


class NumericFunctionValidator:
    """数值函数校验器

    校验数值函数的安全性。
    """

    # 精度损失函数（可能导致信息泄露）
    PRECISION_LOSS_FUNCTIONS = {
        NumericFunctionType.ROUND,
        NumericFunctionType.FLOOR,
        NumericFunctionType.CEIL,
        NumericFunctionType.CEILING,
        NumericFunctionType.TRUNCATE,
        NumericFunctionType.TRUNC,
    }

    # 数学变换函数（可能被用于推断原值）
    MATH_TRANSFORM_FUNCTIONS = {
        NumericFunctionType.LOG,
        NumericFunctionType.LOG10,
        NumericFunctionType.LOG2,
        NumericFunctionType.LN,
        NumericFunctionType.SQRT,
        NumericFunctionType.POWER,
        NumericFunctionType.POW,
        NumericFunctionType.EXP,
    }

    # 类型转换函数
    TYPE_CAST_FUNCTIONS = {
        NumericFunctionType.CAST,
        NumericFunctionType.CONVERT,
        NumericFunctionType.TO_NUMBER,
        NumericFunctionType.TO_DECIMAL,
        NumericFunctionType.TO_FLOAT,
        NumericFunctionType.TO_DOUBLE,
    }

    # 比较函数（可能泄露极值）
    COMPARISON_FUNCTIONS = {
        NumericFunctionType.GREATEST,
        NumericFunctionType.LEAST,
    }

    # 随机函数
    RANDOM_FUNCTIONS = {
        NumericFunctionType.RAND,
        NumericFunctionType.RANDOM,
    }

    def __init__(self, get_field_security_func=None):
        """初始化校验器

        Args:
            get_field_security_func: 获取字段安全配置的函数
        """
        self.get_field_security = get_field_security_func

    def validate(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """校验数值函数调用

        Args:
            func_call: 数值函数调用

        Returns:
            问题列表
        """
        issues = []

        # 检查精度损失函数
        if func_call.func_type in self.PRECISION_LOSS_FUNCTIONS:
            issues.extend(self._check_precision_loss_function(func_call))

        # 检查数学变换函数
        if func_call.func_type in self.MATH_TRANSFORM_FUNCTIONS:
            issues.extend(self._check_math_transform_function(func_call))

        # 检查类型转换函数
        if func_call.func_type in self.TYPE_CAST_FUNCTIONS:
            issues.extend(self._check_type_cast_function(func_call))

        # 检查比较函数
        if func_call.func_type in self.COMPARISON_FUNCTIONS:
            issues.extend(self._check_comparison_function(func_call))

        # 检查随机函数
        if func_call.func_type in self.RANDOM_FUNCTIONS:
            issues.extend(self._check_random_function(func_call))

        # 检查涉及敏感字段的函数
        issues.extend(self._check_sensitive_columns(func_call))

        return issues

    def _check_precision_loss_function(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """检查精度损失函数"""
        issues = []

        # ROUND可以用于隐私保护，但需要检查精度
        if func_call.func_type == NumericFunctionType.ROUND:
            if func_call.precision is not None and func_call.precision > 2:
                issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                    risk_level=RiskLevel.LOW,
                    description=f"ROUND函数保留了较高精度 ({func_call.precision}位小数)",
                    sql_fragment=func_call.raw_text,
                    remediation="考虑降低精度以增强隐私保护",
                    affected_fields=func_call.column_refs,
                    validator_name="NumericFunctionValidator"
                ))
        else:
            # FLOOR/CEIL可能用于推断原值范围
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"函数 {func_call.func_name}() 可能泄露数值边界信息",
                sql_fragment=func_call.raw_text,
                remediation="注意取整操作可能被用于推断原值范围",
                affected_fields=func_call.column_refs,
                validator_name="NumericFunctionValidator"
            ))

        return issues

    def _check_math_transform_function(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """检查数学变换函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.LOW,
            description=f"数学变换函数 {func_call.func_name}() 的结果可能被逆向推算",
            sql_fragment=func_call.raw_text,
            remediation="数学变换通常是可逆的，注意保护敏感字段",
            affected_fields=func_call.column_refs,
            validator_name="NumericFunctionValidator"
        ))

        return issues

    def _check_type_cast_function(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """检查类型转换函数"""
        issues = []

        # 检查是否转换为高精度类型
        high_precision_types = {'DECIMAL', 'NUMERIC', 'DOUBLE', 'FLOAT'}
        if func_call.target_type and \
           any(t in func_call.target_type.upper() for t in high_precision_types):
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
                risk_level=RiskLevel.LOW,
                description=f"类型转换 {func_call.func_name}() 保留了高精度",
                sql_fragment=func_call.raw_text,
                remediation="考虑转换为较低精度类型以增强隐私保护",
                affected_fields=func_call.column_refs,
                validator_name="NumericFunctionValidator"
            ))

        return issues

    def _check_comparison_function(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """检查比较函数"""
        issues = []

        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.AGGREGATE_DISCLOSURE,
            risk_level=RiskLevel.MEDIUM,
            description=f"比较函数 {func_call.func_name}() 可能泄露极值信息",
            sql_fragment=func_call.raw_text,
            remediation="GREATEST/LEAST可能暴露字段的最大/最小值",
            affected_fields=func_call.column_refs,
            validator_name="NumericFunctionValidator"
        ))

        return issues

    def _check_random_function(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
        """检查随机函数"""
        issues = []

        # 随机函数用于差分隐私是好的，但可能影响可重复性
        issues.append(ValidationIssue(
            issue_type=ValidationIssueType.UNKNOWN_RISK,
            risk_level=RiskLevel.LOW,
            description=f"随机函数 {func_call.func_name}() 的使用可能影响结果可重复性",
            sql_fragment=func_call.raw_text,
            remediation="确保随机数用途正确（如差分隐私噪声）",
            affected_fields=func_call.column_refs,
            validator_name="NumericFunctionValidator"
        ))

        return issues

    def _check_sensitive_columns(self, func_call: NumericFunctionCall) -> List[ValidationIssue]:
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
                    # 检查是否是安全的数值操作（如ROUND用于泛化）
                    safe_operations = {
                        NumericFunctionType.ROUND,
                        NumericFunctionType.FLOOR,
                        NumericFunctionType.CEIL,
                    }
                    if func_call.func_type not in safe_operations:
                        issues.append(ValidationIssue(
                            issue_type=ValidationIssueType.DIRECT_FIELD_EXPOSURE,
                            risk_level=RiskLevel.HIGH,
                            description=f"数值函数 {func_call.func_name}() 操作敏感数值字段 {col_ref}",
                            sql_fragment=func_call.raw_text,
                            remediation="敏感数值字段应使用ROUND进行泛化处理",
                            affected_fields=[col_ref],
                            validator_name="NumericFunctionValidator"
                        ))
            except Exception:
                pass

        return issues


# 数值函数安全规则
NUMERIC_FUNCTION_RULES = {
    NumericFunctionType.ABS: {
        "risk_level": RiskLevel.SAFE,
        "description": "取绝对值，通常安全"
    },
    NumericFunctionType.ROUND: {
        "risk_level": RiskLevel.SAFE,
        "description": "四舍五入，可用于隐私保护"
    },
    NumericFunctionType.FLOOR: {
        "risk_level": RiskLevel.LOW,
        "description": "向下取整，可能泄露范围信息"
    },
    NumericFunctionType.CEIL: {
        "risk_level": RiskLevel.LOW,
        "description": "向上取整，可能泄露范围信息"
    },
    NumericFunctionType.SQRT: {
        "risk_level": RiskLevel.LOW,
        "description": "平方根，可逆变换"
    },
    NumericFunctionType.LOG: {
        "risk_level": RiskLevel.LOW,
        "description": "对数变换，可逆"
    },
    NumericFunctionType.POWER: {
        "risk_level": RiskLevel.LOW,
        "description": "幂运算，可能可逆"
    },
    NumericFunctionType.CAST: {
        "risk_level": RiskLevel.LOW,
        "description": "类型转换，取决于目标类型"
    },
    NumericFunctionType.GREATEST: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "返回最大值，可能泄露极值"
    },
    NumericFunctionType.LEAST: {
        "risk_level": RiskLevel.MEDIUM,
        "description": "返回最小值，可能泄露极值"
    },
    NumericFunctionType.RAND: {
        "risk_level": RiskLevel.SAFE,
        "description": "随机数生成，通常安全"
    },
    NumericFunctionType.MOD: {
        "risk_level": RiskLevel.LOW,
        "description": "取模运算，可能泄露余数信息"
    },
}


def get_numeric_function_risk(func_type: NumericFunctionType) -> RiskLevel:
    """获取数值函数的风险等级"""
    rule = NUMERIC_FUNCTION_RULES.get(func_type)
    if rule:
        return rule["risk_level"]
    return RiskLevel.LOW


def is_safe_numeric_function(func_name: str) -> bool:
    """检查数值函数是否安全"""
    func_type = NumericFunctionType.from_string(func_name)
    return get_numeric_function_risk(func_type) == RiskLevel.SAFE
