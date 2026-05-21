"""
SQL Processing Base Classes
SQL处理基础类

提供SQL查询执行和验证功能。
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SQLEngine:
    """
    SQL执行引擎

    使用pandas/SQLite执行SQL查询。
    """

    def __init__(self, backend: str = 'sqlite'):
        """
        初始化SQL引擎

        Args:
            backend: 执行后端（sqlite, duckdb, pandasql）
        """
        self.backend = backend
        self._connection = None
        self._registered_tables = {}

    def register_table(self, name: str, data: pd.DataFrame):
        """
        注册数据表

        Args:
            name: 表名
            data: 数据
        """
        self._registered_tables[name] = data
        logger.info(f"Registered table: {name}, shape={data.shape}")

    def execute(self, query: str) -> pd.DataFrame:
        """
        执行SQL查询

        Args:
            query: SQL查询语句

        Returns:
            查询结果
        """
        logger.info(f"Executing SQL query: {query[:100]}...")

        if self.backend == 'sqlite':
            return self._execute_sqlite(query)
        elif self.backend == 'duckdb':
            return self._execute_duckdb(query)
        elif self.backend == 'pandasql':
            return self._execute_pandasql(query)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _execute_sqlite(self, query: str) -> pd.DataFrame:
        """使用SQLite执行查询"""
        import sqlite3

        conn = sqlite3.connect(':memory:')

        try:
            # 将DataFrame写入SQLite
            for name, df in self._registered_tables.items():
                df.to_sql(name, conn, index=False, if_exists='replace')

            # 执行查询
            result = pd.read_sql_query(query, conn)
            return result

        finally:
            conn.close()

    def _execute_duckdb(self, query: str) -> pd.DataFrame:
        """使用DuckDB执行查询"""
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required for duckdb backend")

        conn = duckdb.connect(':memory:')

        try:
            # 注册表
            for name, df in self._registered_tables.items():
                conn.register(name, df)

            # 执行查询
            result = conn.execute(query).fetchdf()
            return result

        finally:
            conn.close()

    def _execute_pandasql(self, query: str) -> pd.DataFrame:
        """使用pandasql执行查询"""
        try:
            from pandasql import sqldf
        except ImportError:
            raise ImportError("pandasql is required for pandasql backend")

        # pandasql需要本地变量环境
        local_env = self._registered_tables.copy()
        result = sqldf(query, local_env)
        return result

    def get_table_info(self) -> Dict[str, Dict]:
        """获取注册的表信息"""
        info = {}
        for name, df in self._registered_tables.items():
            info[name] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }
        return info


class SQLValidator:
    """
    SQL验证器

    验证SQL查询的安全性和语法。
    """

    # 危险的SQL模式
    DANGEROUS_PATTERNS = [
        r'\bDROP\s+TABLE\b',
        r'\bDROP\s+DATABASE\b',
        r'\bTRUNCATE\b',
        r'\bDELETE\s+FROM\b',
        r'\bINSERT\s+INTO\b',
        r'\bUPDATE\s+.*\s+SET\b',
        r'\bALTER\s+TABLE\b',
        r'\bCREATE\s+TABLE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'--',  # SQL注释（可能用于注入）
        r'/\*',  # 多行注释开始
    ]

    ALLOWED_KEYWORDS = [
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT',
        'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL',
        'ORDER', 'BY', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
        'GROUP', 'HAVING', 'AS', 'DISTINCT', 'ALL',
        'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'ON',
        'UNION', 'INTERSECT', 'EXCEPT',
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'CAST', 'COALESCE', 'NULLIF', 'IFNULL',
        'UPPER', 'LOWER', 'LENGTH', 'SUBSTR', 'TRIM',
        'ROUND', 'ABS', 'FLOOR', 'CEIL',
        'DATE', 'TIME', 'DATETIME', 'STRFTIME',
        'WITH',  # CTE支持
    ]

    def __init__(self, allow_modifications: bool = False):
        """
        初始化验证器

        Args:
            allow_modifications: 是否允许修改操作
        """
        self.allow_modifications = allow_modifications

    def validate(self, query: str) -> Dict[str, Any]:
        """
        验证SQL查询

        Args:
            query: SQL查询语句

        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        query_upper = query.upper()

        # 检查危险模式
        if not self.allow_modifications:
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, query_upper, re.IGNORECASE):
                    result["valid"] = False
                    result["errors"].append(f"Dangerous SQL pattern detected: {pattern}")

        # 基本语法检查
        if not query_upper.strip().startswith('SELECT') and not query_upper.strip().startswith('WITH'):
            if not self.allow_modifications:
                result["valid"] = False
                result["errors"].append("Only SELECT queries are allowed")

        # 检查括号匹配
        if query.count('(') != query.count(')'):
            result["warnings"].append("Unmatched parentheses")

        return result


class SQLQueryBuilder:
    """
    SQL查询构建器

    辅助构建SQL查询。
    """

    def __init__(self, table_name: str):
        """
        初始化查询构建器

        Args:
            table_name: 主表名
        """
        self._table = table_name
        self._select = ['*']
        self._where = []
        self._group_by = []
        self._having = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._joins = []

    def select(self, *columns: str) -> 'SQLQueryBuilder':
        """设置SELECT列"""
        self._select = list(columns)
        return self

    def where(self, condition: str) -> 'SQLQueryBuilder':
        """添加WHERE条件"""
        self._where.append(condition)
        return self

    def join(self, table: str, on: str, join_type: str = 'INNER') -> 'SQLQueryBuilder':
        """添加JOIN"""
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def group_by(self, *columns: str) -> 'SQLQueryBuilder':
        """设置GROUP BY"""
        self._group_by = list(columns)
        return self

    def having(self, condition: str) -> 'SQLQueryBuilder':
        """添加HAVING条件"""
        self._having.append(condition)
        return self

    def order_by(self, column: str, desc: bool = False) -> 'SQLQueryBuilder':
        """设置ORDER BY"""
        order = "DESC" if desc else "ASC"
        self._order_by.append(f"{column} {order}")
        return self

    def limit(self, n: int) -> 'SQLQueryBuilder':
        """设置LIMIT"""
        self._limit = n
        return self

    def offset(self, n: int) -> 'SQLQueryBuilder':
        """设置OFFSET"""
        self._offset = n
        return self

    def build(self) -> str:
        """构建SQL查询"""
        parts = []

        # SELECT
        parts.append(f"SELECT {', '.join(self._select)}")

        # FROM
        parts.append(f"FROM {self._table}")

        # JOINs
        if self._joins:
            parts.extend(self._joins)

        # WHERE
        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")

        # GROUP BY
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        # HAVING
        if self._having:
            parts.append(f"HAVING {' AND '.join(self._having)}")

        # ORDER BY
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        # LIMIT
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        # OFFSET
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        return ' '.join(parts)
