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
SQL格式化模块

提供联邦分析场景下的SQL格式化、美化和标准化功能。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

try:
    import sqlparse
    from sqlparse import tokens as T
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False


class FormatStyle(Enum):
    """格式化风格"""
    COMPACT = "compact"         # 紧凑格式
    READABLE = "readable"       # 可读格式
    EXPANDED = "expanded"       # 展开格式（每个子句独立一行）


class KeywordCase(Enum):
    """关键字大小写"""
    UPPER = "upper"
    LOWER = "lower"
    CAPITALIZE = "capitalize"


class IdentifierCase(Enum):
    """标识符大小写"""
    UPPER = "upper"
    LOWER = "lower"
    UNCHANGED = "unchanged"


@dataclass
class FormatOptions:
    """格式化选项"""
    # 基本格式
    style: FormatStyle = FormatStyle.READABLE
    indent_width: int = 4
    indent_char: str = " "

    # 大小写
    keyword_case: KeywordCase = KeywordCase.UPPER
    identifier_case: IdentifierCase = IdentifierCase.UNCHANGED

    # 换行
    max_line_length: int = 80
    newline_before_from: bool = True
    newline_before_where: bool = True
    newline_before_join: bool = True
    newline_before_group_by: bool = True
    newline_before_order_by: bool = True
    newline_before_having: bool = True
    newline_before_limit: bool = True

    # 逗号
    comma_first: bool = False       # 逗号在前
    space_after_comma: bool = True

    # 括号
    space_around_operators: bool = True
    space_after_open_paren: bool = False
    space_before_close_paren: bool = False

    # 其他
    strip_comments: bool = False
    strip_whitespace: bool = True
    reindent: bool = True


class SQLFormatter:
    """SQL格式化器

    提供SQL语句的格式化和美化功能。
    """

    # SQL关键字列表
    KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'EXISTS',
        'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS', 'ON',
        'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
        'UNION', 'INTERSECT', 'EXCEPT', 'ALL', 'DISTINCT',
        'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE',
        'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX', 'VIEW',
        'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'BETWEEN', 'LIKE',
        'IS', 'NULL', 'TRUE', 'FALSE', 'WITH', 'RECURSIVE'
    }

    # 主要子句（需要换行的）
    MAIN_CLAUSES = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'HAVING',
                    'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'}

    def __init__(self, options: FormatOptions = None):
        """初始化格式化器

        Args:
            options: 格式化选项
        """
        self.options = options or FormatOptions()

    def format(self, sql: str) -> str:
        """格式化SQL语句

        Args:
            sql: SQL语句

        Returns:
            格式化后的SQL
        """
        if HAS_SQLPARSE:
            return self._format_with_sqlparse(sql)
        else:
            return self._format_simple(sql)

    def _format_with_sqlparse(self, sql: str) -> str:
        """使用sqlparse进行格式化"""
        # 构建sqlparse格式化参数
        format_kwargs = {
            'reindent': self.options.reindent,
            'indent_width': self.options.indent_width,
            'keyword_case': self.options.keyword_case.value,
            'strip_comments': self.options.strip_comments,
            'strip_whitespace': self.options.strip_whitespace,
        }

        # 根据格式化风格调整参数
        if self.options.style == FormatStyle.COMPACT:
            format_kwargs['reindent'] = False
        elif self.options.style == FormatStyle.EXPANDED:
            format_kwargs['reindent'] = True
            format_kwargs['indent_width'] = self.options.indent_width

        formatted = sqlparse.format(sql, **format_kwargs)

        # 应用额外的格式化规则
        formatted = self._apply_additional_rules(formatted)

        return formatted

    def _format_simple(self, sql: str) -> str:
        """简单格式化（不依赖sqlparse）"""
        # 标准化空白
        sql = ' '.join(sql.split())

        # 应用关键字大小写
        sql = self._apply_keyword_case(sql)

        # 应用标识符大小写
        sql = self._apply_identifier_case(sql)

        # 添加换行
        sql = self._add_newlines(sql)

        # 添加缩进
        sql = self._add_indentation(sql)

        return sql

    def _apply_additional_rules(self, sql: str) -> str:
        """应用额外的格式化规则"""
        # 处理逗号
        if self.options.comma_first:
            sql = self._apply_comma_first(sql)

        # 处理操作符周围的空格
        if self.options.space_around_operators:
            sql = self._apply_operator_spacing(sql)

        return sql

    def _apply_keyword_case(self, sql: str) -> str:
        """应用关键字大小写"""
        for keyword in self.KEYWORDS:
            pattern = rf'\b{keyword}\b'
            if self.options.keyword_case == KeywordCase.UPPER:
                replacement = keyword.upper()
            elif self.options.keyword_case == KeywordCase.LOWER:
                replacement = keyword.lower()
            else:
                replacement = keyword.capitalize()
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        return sql

    def _apply_identifier_case(self, sql: str) -> str:
        """应用标识符大小写"""
        if self.options.identifier_case == IdentifierCase.UNCHANGED:
            return sql

        # 简单处理：不改变引号内的标识符
        # 这是一个简化实现
        return sql

    def _add_newlines(self, sql: str) -> str:
        """添加换行"""
        if self.options.style == FormatStyle.COMPACT:
            return sql

        # 在主要子句前添加换行
        if self.options.newline_before_from:
            sql = re.sub(r'\bFROM\b', '\nFROM', sql, flags=re.IGNORECASE)
        if self.options.newline_before_where:
            sql = re.sub(r'\bWHERE\b', '\nWHERE', sql, flags=re.IGNORECASE)
        if self.options.newline_before_join:
            sql = re.sub(r'\b(LEFT|RIGHT|INNER|OUTER|CROSS|FULL)?\s*JOIN\b',
                         r'\n\1 JOIN', sql, flags=re.IGNORECASE)
        if self.options.newline_before_group_by:
            sql = re.sub(r'\bGROUP\s+BY\b', '\nGROUP BY', sql, flags=re.IGNORECASE)
        if self.options.newline_before_order_by:
            sql = re.sub(r'\bORDER\s+BY\b', '\nORDER BY', sql, flags=re.IGNORECASE)
        if self.options.newline_before_having:
            sql = re.sub(r'\bHAVING\b', '\nHAVING', sql, flags=re.IGNORECASE)
        if self.options.newline_before_limit:
            sql = re.sub(r'\bLIMIT\b', '\nLIMIT', sql, flags=re.IGNORECASE)

        return sql

    def _add_indentation(self, sql: str) -> str:
        """添加缩进"""
        if self.options.style == FormatStyle.COMPACT:
            return sql

        lines = sql.split('\n')
        formatted_lines = []
        indent = self.options.indent_char * self.options.indent_width

        for line in lines:
            line = line.strip()
            if line:
                # 主要子句不缩进，其他内容缩进
                if any(line.upper().startswith(clause) for clause in
                       ('SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY',
                        'HAVING', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT')):
                    formatted_lines.append(line)
                elif line.upper().startswith(('JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS')):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(indent + line)

        return '\n'.join(formatted_lines)

    def _apply_comma_first(self, sql: str) -> str:
        """应用逗号在前风格"""
        # 简单实现：将 ", " 替换为换行加逗号
        # 这需要更复杂的解析来正确处理
        return sql

    def _apply_operator_spacing(self, sql: str) -> str:
        """应用操作符周围的空格"""
        # 添加操作符周围的空格
        operators = ['=', '<>', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/']
        for op in operators:
            # 避免影响字符串内的操作符
            sql = re.sub(rf'(?<=[^\s]){re.escape(op)}(?=[^\s])',
                         f' {op} ', sql)
        return sql


class SQLNormalizer:
    """SQL标准化器

    将SQL标准化为统一格式，便于比较和分析。
    """

    def __init__(self):
        """初始化标准化器"""
        self.formatter = SQLFormatter(FormatOptions(
            style=FormatStyle.COMPACT,
            keyword_case=KeywordCase.UPPER,
            identifier_case=IdentifierCase.LOWER,
            strip_comments=True,
            strip_whitespace=True
        ))

    def normalize(self, sql: str) -> str:
        """标准化SQL语句

        Args:
            sql: SQL语句

        Returns:
            标准化后的SQL
        """
        # 移除注释
        sql = self._remove_comments(sql)

        # 标准化空白
        sql = ' '.join(sql.split())

        # 标准化关键字
        sql = self._normalize_keywords(sql)

        # 标准化标识符
        sql = self._normalize_identifiers(sql)

        # 标准化字面值格式
        sql = self._normalize_literals(sql)

        return sql

    def _remove_comments(self, sql: str) -> str:
        """移除SQL注释"""
        # 移除单行注释
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # 移除多行注释
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql

    def _normalize_keywords(self, sql: str) -> str:
        """标准化关键字为大写"""
        keywords = SQLFormatter.KEYWORDS
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            sql = re.sub(pattern, keyword.upper(), sql, flags=re.IGNORECASE)
        return sql

    def _normalize_identifiers(self, sql: str) -> str:
        """标准化标识符"""
        # 移除不必要的引号
        sql = re.sub(r'`([^`]+)`', r'\1', sql)
        sql = re.sub(r'"([^"]+)"', r'\1', sql)
        # 将标识符转换为小写（不在引号内的）
        # 使用简单策略：除关键字外的单词转换为小写
        keywords = SQLFormatter.KEYWORDS
        result = []
        for word in sql.split():
            if word.upper() not in keywords and not word.startswith("'"):
                result.append(word.lower())
            else:
                result.append(word)
        return ' '.join(result)

    def _normalize_literals(self, sql: str) -> str:
        """标准化字面值格式"""
        # 统一使用单引号
        sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
        return sql

    def get_sql_hash(self, sql: str) -> str:
        """获取SQL的哈希值（用于比较）

        Args:
            sql: SQL语句

        Returns:
            哈希值
        """
        import hashlib
        normalized = self.normalize(sql)
        return hashlib.md5(normalized.encode()).hexdigest()


class SQLPrettifier:
    """SQL美化器

    提供更高级的SQL美化功能。
    """

    def __init__(self, options: FormatOptions = None):
        """初始化美化器

        Args:
            options: 格式化选项
        """
        self.options = options or FormatOptions(style=FormatStyle.EXPANDED)
        self.formatter = SQLFormatter(self.options)

    def prettify(self, sql: str) -> str:
        """美化SQL语句

        Args:
            sql: SQL语句

        Returns:
            美化后的SQL
        """
        # 基本格式化
        formatted = self.formatter.format(sql)

        # 高级美化
        formatted = self._align_columns(formatted)
        formatted = self._add_clause_spacing(formatted)

        return formatted

    def _align_columns(self, sql: str) -> str:
        """对齐列名"""
        # 简化实现，实际需要解析SELECT列表
        return sql

    def _add_clause_spacing(self, sql: str) -> str:
        """添加子句间距"""
        # 在主要子句之间添加空行
        sql = re.sub(r'\n(FROM|WHERE|GROUP BY|HAVING|ORDER BY)',
                     r'\n\n\1', sql, flags=re.IGNORECASE)
        return sql


class SQLCompactor:
    """SQL压缩器

    将SQL压缩为单行格式。
    """

    def compact(self, sql: str) -> str:
        """压缩SQL语句

        Args:
            sql: SQL语句

        Returns:
            压缩后的SQL
        """
        # 移除注释
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # 标准化空白为单个空格
        sql = ' '.join(sql.split())

        return sql


# 便捷函数

def format_sql(sql: str, style: FormatStyle = FormatStyle.READABLE,
               keyword_case: KeywordCase = KeywordCase.UPPER) -> str:
    """格式化SQL语句

    Args:
        sql: SQL语句
        style: 格式化风格
        keyword_case: 关键字大小写

    Returns:
        格式化后的SQL

    Example:
        formatted = format_sql("select * from users where id=1")
        # 返回:
        # SELECT *
        # FROM users
        # WHERE id = 1
    """
    options = FormatOptions(style=style, keyword_case=keyword_case)
    formatter = SQLFormatter(options)
    return formatter.format(sql)


def normalize_sql(sql: str) -> str:
    """标准化SQL语句

    Args:
        sql: SQL语句

    Returns:
        标准化后的SQL

    Example:
        normalized = normalize_sql("SELECT * FROM users")
        # 返回: "SELECT * FROM users"
    """
    normalizer = SQLNormalizer()
    return normalizer.normalize(sql)


def compact_sql(sql: str) -> str:
    """压缩SQL语句

    Args:
        sql: SQL语句

    Returns:
        压缩后的SQL

    Example:
        compacted = compact_sql('''
            SELECT *
            FROM users
            WHERE id = 1
        ''')
        # 返回: "SELECT * FROM users WHERE id = 1"
    """
    compactor = SQLCompactor()
    return compactor.compact(sql)


def prettify_sql(sql: str) -> str:
    """美化SQL语句

    Args:
        sql: SQL语句

    Returns:
        美化后的SQL
    """
    prettifier = SQLPrettifier()
    return prettifier.prettify(sql)


def compare_sql(sql1: str, sql2: str) -> bool:
    """比较两个SQL语句是否等价

    Args:
        sql1: 第一个SQL语句
        sql2: 第二个SQL语句

    Returns:
        是否等价

    Example:
        same = compare_sql(
            "select * from users",
            "SELECT * FROM users"
        )
        # 返回: True
    """
    normalizer = SQLNormalizer()
    return normalizer.normalize(sql1) == normalizer.normalize(sql2)
