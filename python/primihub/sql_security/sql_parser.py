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
SQL解析封装模块

使用 sqlparse 库解析SQL语句并提取结构化信息。
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

try:
    import sqlparse
    from sqlparse import sql as sql_types
    from sqlparse import tokens as T
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


class AggregateType(Enum):
    """聚合函数类型"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"
    GROUP_CONCAT = "GROUP_CONCAT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, func_name: str) -> 'AggregateType':
        """从函数名转换"""
        name_upper = func_name.upper()
        mapping = {
            'COUNT': cls.COUNT,
            'SUM': cls.SUM,
            'AVG': cls.AVG,
            'MIN': cls.MIN,
            'MAX': cls.MAX,
            'STDDEV': cls.STDDEV,
            'STDEV': cls.STDDEV,
            'VARIANCE': cls.VARIANCE,
            'VAR': cls.VARIANCE,
            'GROUP_CONCAT': cls.GROUP_CONCAT
        }
        return mapping.get(name_upper, cls.UNKNOWN)


class JoinType(Enum):
    """JOIN类型"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    NATURAL = "NATURAL"
    UNKNOWN = "UNKNOWN"


class ComparisonOp(Enum):
    """比较操作符"""
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT EXISTS"


@dataclass
class ColumnRef:
    """列引用"""
    column_name: str
    table_name: str = ""
    alias: str = ""
    raw_text: str = ""

    def full_name(self) -> str:
        """获取完整名称"""
        if self.table_name:
            return f"{self.table_name}.{self.column_name}"
        return self.column_name


@dataclass
class TableRef:
    """表引用"""
    table_name: str
    schema_name: str = ""
    alias: str = ""
    raw_text: str = ""

    def full_name(self) -> str:
        """获取完整名称"""
        if self.schema_name:
            return f"{self.schema_name}.{self.table_name}"
        return self.table_name


@dataclass
class JoinInfo:
    """JOIN信息"""
    join_type: JoinType
    left_table: Optional[TableRef] = None
    right_table: Optional[TableRef] = None
    on_columns: List[Tuple[ColumnRef, ColumnRef]] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Condition:
    """条件表达式"""
    operator: ComparisonOp
    left_operand: Any  # ColumnRef 或字面值
    right_operand: Any  # ColumnRef 或字面值或列表
    raw_text: str = ""
    is_column_ref_left: bool = True

    def get_columns(self) -> List[ColumnRef]:
        """获取涉及的列"""
        columns = []
        if isinstance(self.left_operand, ColumnRef):
            columns.append(self.left_operand)
        if isinstance(self.right_operand, ColumnRef):
            columns.append(self.right_operand)
        return columns


@dataclass
class AggregateCall:
    """聚合函数调用"""
    agg_type: AggregateType
    arguments: List[ColumnRef] = field(default_factory=list)
    distinct: bool = False
    alias: str = ""
    raw_text: str = ""


@dataclass
class WindowCall:
    """窗口函数调用"""
    function_name: str
    arguments: List[ColumnRef] = field(default_factory=list)
    partition_by: List[ColumnRef] = field(default_factory=list)
    order_by: List[Tuple[ColumnRef, bool]] = field(default_factory=list)  # (column, is_asc)
    alias: str = ""
    raw_text: str = ""


@dataclass
class SubqueryInfo:
    """子查询信息"""
    is_correlated: bool = False
    correlated_columns: List[ColumnRef] = field(default_factory=list)
    subquery_type: str = ""  # scalar, in, exists, from
    raw_text: str = ""
    nested_parsed: Optional['ParsedSQL'] = None


@dataclass
class OrderByItem:
    """ORDER BY项"""
    column: ColumnRef
    is_ascending: bool = True
    nulls_first: Optional[bool] = None


@dataclass
class ParsedSQL:
    """解析后的SQL结构

    包含SQL语句的所有结构化信息。
    """
    original_sql: str = ""
    statement_type: str = "SELECT"  # SELECT, INSERT, UPDATE, DELETE

    # SELECT相关
    select_columns: List[ColumnRef] = field(default_factory=list)
    select_all: bool = False  # SELECT *

    # FROM相关
    from_tables: List[TableRef] = field(default_factory=list)
    table_aliases: Dict[str, str] = field(default_factory=dict)

    # JOIN相关
    joins: List[JoinInfo] = field(default_factory=list)

    # WHERE相关
    where_conditions: List[Condition] = field(default_factory=list)
    where_raw: str = ""

    # GROUP BY相关
    group_by_columns: List[ColumnRef] = field(default_factory=list)

    # HAVING相关
    having_conditions: List[Condition] = field(default_factory=list)
    having_raw: str = ""

    # ORDER BY相关
    order_by_items: List[OrderByItem] = field(default_factory=list)

    # LIMIT相关
    limit: Optional[int] = None
    offset: Optional[int] = None

    # 聚合函数
    aggregates: List[AggregateCall] = field(default_factory=list)

    # 窗口函数
    window_functions: List[WindowCall] = field(default_factory=list)

    # 子查询
    subqueries: List[SubqueryInfo] = field(default_factory=list)

    # 解析状态
    parse_error: Optional[str] = None
    is_valid: bool = True

    def has_joins(self) -> bool:
        """是否有JOIN"""
        return len(self.joins) > 0

    def has_aggregates(self) -> bool:
        """是否有聚合函数"""
        return len(self.aggregates) > 0

    def has_group_by(self) -> bool:
        """是否有GROUP BY"""
        return len(self.group_by_columns) > 0

    def has_order_by(self) -> bool:
        """是否有ORDER BY"""
        return len(self.order_by_items) > 0

    def has_window_functions(self) -> bool:
        """是否有窗口函数"""
        return len(self.window_functions) > 0

    def has_subqueries(self) -> bool:
        """是否有子查询"""
        return len(self.subqueries) > 0

    def has_correlated_subqueries(self) -> bool:
        """是否有关联子查询"""
        return any(sq.is_correlated for sq in self.subqueries)

    def get_all_tables(self) -> List[str]:
        """获取所有涉及的表名"""
        tables = set()
        for table in self.from_tables:
            tables.add(table.table_name)
        for join in self.joins:
            if join.left_table:
                tables.add(join.left_table.table_name)
            if join.right_table:
                tables.add(join.right_table.table_name)
        return list(tables)

    def get_all_columns(self) -> List[ColumnRef]:
        """获取所有涉及的列"""
        columns = []
        columns.extend(self.select_columns)
        columns.extend(self.group_by_columns)
        for item in self.order_by_items:
            columns.append(item.column)
        for cond in self.where_conditions:
            columns.extend(cond.get_columns())
        for agg in self.aggregates:
            columns.extend(agg.arguments)
        return columns


class SQLParser:
    """SQL解析器

    封装 sqlparse 库，提供结构化的SQL解析功能。
    """

    # 聚合函数名列表
    AGGREGATE_FUNCTIONS = {
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'STDDEV', 'STDEV', 'VARIANCE', 'VAR',
        'GROUP_CONCAT', 'STRING_AGG'
    }

    # 窗口函数名列表
    WINDOW_FUNCTIONS = {
        'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
        'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
        'NTH_VALUE', 'CUME_DIST', 'PERCENT_RANK'
    }

    def __init__(self):
        if not SQLPARSE_AVAILABLE:
            raise ImportError("sqlparse library is required. Install with: pip install sqlparse")

    def parse(self, sql: str) -> ParsedSQL:
        """解析SQL语句

        Args:
            sql: SQL语句

        Returns:
            ParsedSQL 实例
        """
        result = ParsedSQL(original_sql=sql)

        try:
            # 使用sqlparse解析
            statements = sqlparse.parse(sql)
            if not statements:
                result.parse_error = "Empty SQL statement"
                result.is_valid = False
                return result

            stmt = statements[0]
            result.statement_type = stmt.get_type() or "SELECT"

            # 解析各个部分
            self._parse_statement(stmt, result)

        except Exception as e:
            result.parse_error = str(e)
            result.is_valid = False

        return result

    def _parse_statement(self, stmt, result: ParsedSQL) -> None:
        """解析SQL语句"""
        # 标记化分析
        tokens = list(stmt.flatten())

        # 提取SELECT列
        self._extract_select_columns(stmt, result)

        # 提取FROM表
        self._extract_from_tables(stmt, result)

        # 提取JOIN
        self._extract_joins(stmt, result)

        # 提取WHERE条件
        self._extract_where(stmt, result)

        # 提取GROUP BY
        self._extract_group_by(stmt, result)

        # 提取HAVING
        self._extract_having(stmt, result)

        # 提取ORDER BY
        self._extract_order_by(stmt, result)

        # 提取LIMIT
        self._extract_limit(stmt, result)

        # 提取聚合函数
        self._extract_aggregates(stmt, result)

        # 提取窗口函数
        self._extract_window_functions(stmt, result)

        # 提取子查询
        self._extract_subqueries(stmt, result)

    def _extract_select_columns(self, stmt, result: ParsedSQL) -> None:
        """提取SELECT列"""
        select_seen = False

        for token in stmt.tokens:
            if token.ttype is T.DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue

            if select_seen:
                if token.ttype is T.Keyword:
                    break

                if isinstance(token, sql_types.IdentifierList):
                    for identifier in token.get_identifiers():
                        col = self._parse_identifier_as_column(identifier)
                        if col:
                            result.select_columns.append(col)
                elif isinstance(token, sql_types.Identifier):
                    col = self._parse_identifier_as_column(token)
                    if col:
                        result.select_columns.append(col)
                elif token.ttype is T.Wildcard:
                    result.select_all = True

    def _extract_from_tables(self, stmt, result: ParsedSQL) -> None:
        """提取FROM表"""
        from_seen = False

        for token in stmt.tokens:
            if token.ttype is T.Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue

            if from_seen:
                if token.ttype is T.Keyword and token.value.upper() in (
                    'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING',
                    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'FULL'
                ):
                    break

                if isinstance(token, sql_types.IdentifierList):
                    for identifier in token.get_identifiers():
                        table = self._parse_identifier_as_table(identifier)
                        if table:
                            result.from_tables.append(table)
                            if table.alias:
                                result.table_aliases[table.alias] = table.table_name
                elif isinstance(token, sql_types.Identifier):
                    table = self._parse_identifier_as_table(token)
                    if table:
                        result.from_tables.append(table)
                        if table.alias:
                            result.table_aliases[table.alias] = table.table_name

    def _extract_joins(self, stmt, result: ParsedSQL) -> None:
        """提取JOIN信息"""
        sql_upper = stmt.value.upper()

        # 使用正则表达式匹配JOIN
        join_pattern = r'(LEFT|RIGHT|INNER|FULL|CROSS|NATURAL)?\s*(?:OUTER\s+)?JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+([^WHERE|GROUP|ORDER|LIMIT|HAVING]+)'

        for match in re.finditer(join_pattern, stmt.value, re.IGNORECASE):
            join_type_str = match.group(1) or 'INNER'
            table_name = match.group(2)
            table_alias = match.group(3)
            on_clause = match.group(4)

            join_type = JoinType.INNER
            if join_type_str:
                join_type_map = {
                    'LEFT': JoinType.LEFT,
                    'RIGHT': JoinType.RIGHT,
                    'INNER': JoinType.INNER,
                    'FULL': JoinType.FULL,
                    'CROSS': JoinType.CROSS,
                    'NATURAL': JoinType.NATURAL
                }
                join_type = join_type_map.get(join_type_str.upper(), JoinType.INNER)

            join_info = JoinInfo(
                join_type=join_type,
                right_table=TableRef(table_name=table_name, alias=table_alias or ""),
                raw_text=match.group(0)
            )

            # 解析ON条件中的列
            self._parse_join_on_clause(on_clause, join_info)

            result.joins.append(join_info)

    def _parse_join_on_clause(self, on_clause: str, join_info: JoinInfo) -> None:
        """解析JOIN ON子句"""
        # 简单解析 a.col = b.col 格式
        eq_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        for match in re.finditer(eq_pattern, on_clause):
            left_col = ColumnRef(
                table_name=match.group(1),
                column_name=match.group(2)
            )
            right_col = ColumnRef(
                table_name=match.group(3),
                column_name=match.group(4)
            )
            join_info.on_columns.append((left_col, right_col))

    def _extract_where(self, stmt, result: ParsedSQL) -> None:
        """提取WHERE条件"""
        for token in stmt.tokens:
            if isinstance(token, sql_types.Where):
                result.where_raw = str(token)
                self._parse_conditions(token, result.where_conditions)
                break

    def _extract_group_by(self, stmt, result: ParsedSQL) -> None:
        """提取GROUP BY"""
        group_by_seen = False
        by_seen = False

        for token in stmt.tokens:
            if token.ttype is T.Keyword and token.value.upper() == 'GROUP':
                group_by_seen = True
                continue

            if group_by_seen and token.ttype is T.Keyword and token.value.upper() == 'BY':
                by_seen = True
                continue

            if by_seen:
                if token.ttype is T.Keyword and token.value.upper() in (
                    'HAVING', 'ORDER', 'LIMIT'
                ):
                    break

                if isinstance(token, sql_types.IdentifierList):
                    for identifier in token.get_identifiers():
                        col = self._parse_identifier_as_column(identifier)
                        if col:
                            result.group_by_columns.append(col)
                elif isinstance(token, sql_types.Identifier):
                    col = self._parse_identifier_as_column(token)
                    if col:
                        result.group_by_columns.append(col)

    def _extract_having(self, stmt, result: ParsedSQL) -> None:
        """提取HAVING条件"""
        sql_str = str(stmt)
        having_match = re.search(r'HAVING\s+(.+?)(?:ORDER|LIMIT|$)', sql_str, re.IGNORECASE)
        if having_match:
            result.having_raw = having_match.group(1).strip()

    def _extract_order_by(self, stmt, result: ParsedSQL) -> None:
        """提取ORDER BY"""
        order_by_seen = False
        by_seen = False

        for token in stmt.tokens:
            if token.ttype is T.Keyword and token.value.upper() == 'ORDER':
                order_by_seen = True
                continue

            if order_by_seen and token.ttype is T.Keyword and token.value.upper() == 'BY':
                by_seen = True
                continue

            if by_seen:
                if token.ttype is T.Keyword and token.value.upper() == 'LIMIT':
                    break

                if isinstance(token, sql_types.IdentifierList):
                    for identifier in token.get_identifiers():
                        item = self._parse_order_by_item(identifier)
                        if item:
                            result.order_by_items.append(item)
                elif isinstance(token, sql_types.Identifier):
                    item = self._parse_order_by_item(token)
                    if item:
                        result.order_by_items.append(item)

    def _extract_limit(self, stmt, result: ParsedSQL) -> None:
        """提取LIMIT"""
        sql_str = str(stmt)
        limit_match = re.search(r'LIMIT\s+(\d+)(?:\s*,\s*(\d+)|\s+OFFSET\s+(\d+))?',
                                sql_str, re.IGNORECASE)
        if limit_match:
            result.limit = int(limit_match.group(1))
            if limit_match.group(2):
                result.offset = int(limit_match.group(2))
            elif limit_match.group(3):
                result.offset = int(limit_match.group(3))

    def _extract_aggregates(self, stmt, result: ParsedSQL) -> None:
        """提取聚合函数"""
        sql_str = str(stmt)

        for func_name in self.AGGREGATE_FUNCTIONS:
            # 匹配 FUNC(DISTINCT? column) 模式
            pattern = rf'{func_name}\s*\(\s*(DISTINCT\s+)?([^)]+)\)'
            for match in re.finditer(pattern, sql_str, re.IGNORECASE):
                distinct = bool(match.group(1))
                args_str = match.group(2).strip()

                agg = AggregateCall(
                    agg_type=AggregateType.from_string(func_name),
                    distinct=distinct,
                    raw_text=match.group(0)
                )

                # 如果是COUNT DISTINCT，设置特殊类型
                if func_name.upper() == 'COUNT' and distinct:
                    agg.agg_type = AggregateType.COUNT_DISTINCT

                # 解析参数列
                if args_str != '*':
                    col = self._parse_column_string(args_str)
                    if col:
                        agg.arguments.append(col)

                result.aggregates.append(agg)

    def _extract_window_functions(self, stmt, result: ParsedSQL) -> None:
        """提取窗口函数"""
        sql_str = str(stmt)

        # 匹配 FUNC() OVER (PARTITION BY ... ORDER BY ...) 模式
        pattern = r'(\w+)\s*\([^)]*\)\s+OVER\s*\(\s*(PARTITION\s+BY\s+[^)]+)?(?:ORDER\s+BY\s+[^)]+)?\)'

        for match in re.finditer(pattern, sql_str, re.IGNORECASE):
            func_name = match.group(1).upper()
            if func_name in self.WINDOW_FUNCTIONS or func_name in self.AGGREGATE_FUNCTIONS:
                window = WindowCall(
                    function_name=func_name,
                    raw_text=match.group(0)
                )

                # 提取PARTITION BY
                partition_match = re.search(r'PARTITION\s+BY\s+([^ORDER)]+)',
                                           match.group(0), re.IGNORECASE)
                if partition_match:
                    cols_str = partition_match.group(1).strip()
                    for col_str in cols_str.split(','):
                        col = self._parse_column_string(col_str.strip())
                        if col:
                            window.partition_by.append(col)

                result.window_functions.append(window)

    def _extract_subqueries(self, stmt, result: ParsedSQL) -> None:
        """提取子查询"""
        sql_str = str(stmt)

        # 简单匹配括号内的SELECT
        subquery_pattern = r'\(\s*(SELECT\s+.+?)\)'

        for match in re.finditer(subquery_pattern, sql_str, re.IGNORECASE | re.DOTALL):
            subquery_sql = match.group(1)

            # 判断子查询类型
            before_subquery = sql_str[:match.start()].upper()
            subquery_type = "scalar"
            if 'IN' in before_subquery[-10:]:
                subquery_type = "in"
            elif 'EXISTS' in before_subquery[-10:]:
                subquery_type = "exists"
            elif 'FROM' in before_subquery[-10:]:
                subquery_type = "from"

            # 检测是否为关联子查询（简单检测：是否引用外部表）
            outer_tables = [t.table_name for t in result.from_tables]
            outer_aliases = list(result.table_aliases.keys())
            is_correlated = any(
                re.search(rf'\b{t}\b\.', subquery_sql, re.IGNORECASE)
                for t in outer_tables + outer_aliases
            )

            subquery_info = SubqueryInfo(
                is_correlated=is_correlated,
                subquery_type=subquery_type,
                raw_text=subquery_sql
            )

            # 递归解析子查询
            try:
                subquery_info.nested_parsed = self.parse(subquery_sql)
            except Exception:
                pass

            result.subqueries.append(subquery_info)

    def _parse_identifier_as_column(self, identifier) -> Optional[ColumnRef]:
        """解析标识符为列引用"""
        if identifier is None:
            return None

        raw_text = str(identifier).strip()

        # 检查是否是*
        if raw_text == '*':
            return ColumnRef(column_name='*', raw_text=raw_text)

        # 处理别名
        alias = ""
        if isinstance(identifier, sql_types.Identifier):
            alias = identifier.get_alias() or ""

        # 解析 table.column 格式
        name = identifier.get_real_name() if hasattr(identifier, 'get_real_name') else raw_text
        parts = name.split('.') if name else []

        if len(parts) >= 2:
            return ColumnRef(
                table_name=parts[0],
                column_name=parts[1],
                alias=alias,
                raw_text=raw_text
            )
        elif len(parts) == 1:
            return ColumnRef(
                column_name=parts[0],
                alias=alias,
                raw_text=raw_text
            )

        return None

    def _parse_identifier_as_table(self, identifier) -> Optional[TableRef]:
        """解析标识符为表引用"""
        if identifier is None:
            return None

        raw_text = str(identifier).strip()
        alias = ""

        if isinstance(identifier, sql_types.Identifier):
            alias = identifier.get_alias() or ""
            name = identifier.get_real_name() or raw_text
        else:
            name = raw_text

        # 处理 schema.table 格式
        parts = name.split('.')
        if len(parts) >= 2:
            return TableRef(
                schema_name=parts[0],
                table_name=parts[1],
                alias=alias,
                raw_text=raw_text
            )
        else:
            return TableRef(
                table_name=name,
                alias=alias,
                raw_text=raw_text
            )

    def _parse_column_string(self, col_str: str) -> Optional[ColumnRef]:
        """从字符串解析列引用"""
        col_str = col_str.strip()
        if not col_str:
            return None

        parts = col_str.split('.')
        if len(parts) >= 2:
            return ColumnRef(table_name=parts[0], column_name=parts[1])
        else:
            return ColumnRef(column_name=parts[0])

    def _parse_order_by_item(self, identifier) -> Optional[OrderByItem]:
        """解析ORDER BY项"""
        col = self._parse_identifier_as_column(identifier)
        if not col:
            return None

        raw_text = str(identifier).upper()
        is_ascending = 'DESC' not in raw_text

        return OrderByItem(column=col, is_ascending=is_ascending)

    def _parse_conditions(self, where_token, conditions: List[Condition]) -> None:
        """解析WHERE条件"""
        where_str = str(where_token)

        # 移除WHERE关键字
        where_str = re.sub(r'^\s*WHERE\s+', '', where_str, flags=re.IGNORECASE)

        # 解析比较条件
        self._parse_comparison_conditions(where_str, conditions)

    def _parse_comparison_conditions(self, condition_str: str, conditions: List[Condition]) -> None:
        """解析比较条件"""
        # 匹配 column op value 模式
        patterns = [
            # column = value
            (r'(\w+(?:\.\w+)?)\s*=\s*([^\s,)]+)', ComparisonOp.EQ),
            # column != value
            (r'(\w+(?:\.\w+)?)\s*(?:!=|<>)\s*([^\s,)]+)', ComparisonOp.NE),
            # column < value
            (r'(\w+(?:\.\w+)?)\s*<\s*([^\s,)]+)', ComparisonOp.LT),
            # column <= value
            (r'(\w+(?:\.\w+)?)\s*<=\s*([^\s,)]+)', ComparisonOp.LE),
            # column > value
            (r'(\w+(?:\.\w+)?)\s*>\s*([^\s,)]+)', ComparisonOp.GT),
            # column >= value
            (r'(\w+(?:\.\w+)?)\s*>=\s*([^\s,)]+)', ComparisonOp.GE),
            # column LIKE pattern
            (r'(\w+(?:\.\w+)?)\s+LIKE\s+([^\s,)]+)', ComparisonOp.LIKE),
            # column IN (values)
            (r'(\w+(?:\.\w+)?)\s+IN\s*\(([^)]+)\)', ComparisonOp.IN),
            # column BETWEEN value AND value
            (r'(\w+(?:\.\w+)?)\s+BETWEEN\s+([^\s]+)\s+AND\s+([^\s,)]+)', ComparisonOp.BETWEEN),
            # column IS NULL
            (r'(\w+(?:\.\w+)?)\s+IS\s+NULL', ComparisonOp.IS_NULL),
            # column IS NOT NULL
            (r'(\w+(?:\.\w+)?)\s+IS\s+NOT\s+NULL', ComparisonOp.IS_NOT_NULL),
        ]

        for pattern, op in patterns:
            for match in re.finditer(pattern, condition_str, re.IGNORECASE):
                col_str = match.group(1)
                col = self._parse_column_string(col_str)

                right_operand = match.group(2) if len(match.groups()) >= 2 else None

                if op == ComparisonOp.BETWEEN and len(match.groups()) >= 3:
                    # BETWEEN 有两个值
                    right_operand = (match.group(2), match.group(3))

                condition = Condition(
                    operator=op,
                    left_operand=col,
                    right_operand=right_operand,
                    raw_text=match.group(0)
                )
                conditions.append(condition)
