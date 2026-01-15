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
SQL函数安全校验模块

提供各类SQL函数的安全校验功能，包括：
- 字符串函数校验
- 日期函数校验
- 时间戳函数校验
- 数值函数校验

Usage:
    from primihub.sql_security.functions import (
        StringFunctionValidator,
        DateFunctionValidator,
        TimestampFunctionValidator,
        NumericFunctionValidator
    )

    # 提取并校验字符串函数
    extractor = StringFunctionExtractor()
    functions = extractor.extract(sql)

    validator = StringFunctionValidator()
    for func in functions:
        issues = validator.validate(func)
"""

# 字符串函数
from .string_functions import (
    StringFunctionType,
    StringFunctionCall,
    StringFunctionExtractor,
    StringFunctionValidator,
    STRING_FUNCTION_RULES,
    get_string_function_risk,
    is_safe_string_function,
)

# 日期函数
from .date_functions import (
    DateFunctionType,
    DateFunctionCall,
    DateFunctionExtractor,
    DateFunctionValidator,
    DATE_FUNCTION_RULES,
    get_date_function_risk,
    is_safe_date_function,
)

# 时间戳函数
from .timestamp_functions import (
    TimestampFunctionType,
    TimestampFunctionCall,
    TimestampFunctionExtractor,
    TimestampFunctionValidator,
    TIMESTAMP_FUNCTION_RULES,
    get_timestamp_function_risk,
    is_safe_timestamp_function,
)

# 数值函数
from .numeric_functions import (
    NumericFunctionType,
    NumericFunctionCall,
    NumericFunctionExtractor,
    NumericFunctionValidator,
    NUMERIC_FUNCTION_RULES,
    get_numeric_function_risk,
    is_safe_numeric_function,
)


__all__ = [
    # 字符串函数
    'StringFunctionType',
    'StringFunctionCall',
    'StringFunctionExtractor',
    'StringFunctionValidator',
    'STRING_FUNCTION_RULES',
    'get_string_function_risk',
    'is_safe_string_function',

    # 日期函数
    'DateFunctionType',
    'DateFunctionCall',
    'DateFunctionExtractor',
    'DateFunctionValidator',
    'DATE_FUNCTION_RULES',
    'get_date_function_risk',
    'is_safe_date_function',

    # 时间戳函数
    'TimestampFunctionType',
    'TimestampFunctionCall',
    'TimestampFunctionExtractor',
    'TimestampFunctionValidator',
    'TIMESTAMP_FUNCTION_RULES',
    'get_timestamp_function_risk',
    'is_safe_timestamp_function',

    # 数值函数
    'NumericFunctionType',
    'NumericFunctionCall',
    'NumericFunctionExtractor',
    'NumericFunctionValidator',
    'NUMERIC_FUNCTION_RULES',
    'get_numeric_function_risk',
    'is_safe_numeric_function',

    # 聚合功能
    'FunctionExtractor',
    'FunctionValidator',
    'validate_functions',
]


class FunctionExtractor:
    """统一的函数提取器

    从SQL中提取所有类型的函数调用。
    """

    def __init__(self):
        """初始化提取器"""
        self.string_extractor = StringFunctionExtractor()
        self.date_extractor = DateFunctionExtractor()
        self.timestamp_extractor = TimestampFunctionExtractor()
        self.numeric_extractor = NumericFunctionExtractor()

    def extract_all(self, sql: str) -> dict:
        """提取所有函数调用

        Args:
            sql: SQL语句

        Returns:
            包含各类函数调用的字典
        """
        return {
            'string_functions': self.string_extractor.extract(sql),
            'date_functions': self.date_extractor.extract(sql),
            'timestamp_functions': self.timestamp_extractor.extract(sql),
            'numeric_functions': self.numeric_extractor.extract(sql),
        }

    def extract_string_functions(self, sql: str):
        """提取字符串函数"""
        return self.string_extractor.extract(sql)

    def extract_date_functions(self, sql: str):
        """提取日期函数"""
        return self.date_extractor.extract(sql)

    def extract_timestamp_functions(self, sql: str):
        """提取时间戳函数"""
        return self.timestamp_extractor.extract(sql)

    def extract_numeric_functions(self, sql: str):
        """提取数值函数"""
        return self.numeric_extractor.extract(sql)


class FunctionValidator:
    """统一的函数校验器

    校验所有类型的函数调用。
    """

    def __init__(self, get_field_security_func=None):
        """初始化校验器

        Args:
            get_field_security_func: 获取字段安全配置的函数
        """
        self.string_validator = StringFunctionValidator(get_field_security_func)
        self.date_validator = DateFunctionValidator(get_field_security_func)
        self.timestamp_validator = TimestampFunctionValidator(get_field_security_func)
        self.numeric_validator = NumericFunctionValidator(get_field_security_func)
        self.extractor = FunctionExtractor()

    def validate_all(self, sql: str) -> dict:
        """校验SQL中所有函数调用

        Args:
            sql: SQL语句

        Returns:
            包含各类校验结果的字典
        """
        functions = self.extractor.extract_all(sql)
        issues = {
            'string_issues': [],
            'date_issues': [],
            'timestamp_issues': [],
            'numeric_issues': [],
        }

        for func in functions['string_functions']:
            issues['string_issues'].extend(self.string_validator.validate(func))

        for func in functions['date_functions']:
            issues['date_issues'].extend(self.date_validator.validate(func))

        for func in functions['timestamp_functions']:
            issues['timestamp_issues'].extend(self.timestamp_validator.validate(func))

        for func in functions['numeric_functions']:
            issues['numeric_issues'].extend(self.numeric_validator.validate(func))

        return issues

    def get_all_issues(self, sql: str) -> list:
        """获取所有校验问题的扁平列表

        Args:
            sql: SQL语句

        Returns:
            所有问题的列表
        """
        issues_dict = self.validate_all(sql)
        all_issues = []
        for issues in issues_dict.values():
            all_issues.extend(issues)
        return all_issues


def validate_functions(sql: str, get_field_security_func=None) -> list:
    """便捷函数：校验SQL中的所有函数调用

    Args:
        sql: SQL语句
        get_field_security_func: 获取字段安全配置的函数

    Returns:
        校验问题列表

    Example:
        issues = validate_functions(
            "SELECT SUBSTRING(name, 1, 3), YEAR(created_at) FROM users"
        )
        for issue in issues:
            print(f"[{issue.risk_level.name}] {issue.description}")
    """
    validator = FunctionValidator(get_field_security_func)
    return validator.get_all_issues(sql)
