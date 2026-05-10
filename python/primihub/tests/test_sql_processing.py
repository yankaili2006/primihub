import pytest
import pandas as pd
import numpy as np
from python.primihub.local.sql_processing.base import (
    SQLEngine,
    SQLValidator,
    SQLQueryBuilder,
)


class TestSQLEngine:
    def test_register_and_query(self):
        engine = SQLEngine(backend="sqlite")
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        engine.register_table("test_table", data)
        result = engine.execute("SELECT * FROM test_table")
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_select_columns(self):
        engine = SQLEngine(backend="sqlite")
        data = pd.DataFrame({"x": [10, 20], "y": [30, 40], "z": [50, 60]})
        engine.register_table("t2", data)
        result = engine.execute("SELECT x, z FROM t2")
        assert list(result.columns) == ["x", "z"]

    def test_aggregation(self):
        engine = SQLEngine(backend="sqlite")
        data = pd.DataFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
        engine.register_table("t3", data)
        result = engine.execute("SELECT cat, SUM(val) as s FROM t3 GROUP BY cat")
        assert len(result) == 2

    def test_get_table_info(self):
        engine = SQLEngine(backend="sqlite")
        data = pd.DataFrame({"a": [1], "b": ["x"]})
        engine.register_table("t4", data)
        info = engine.get_table_info()
        assert "t4" in info


class TestSQLValidator:
    def test_valid_select(self):
        validator = SQLValidator()
        result = validator.validate("SELECT 1")
        assert result["valid"]

    def test_dangerous_pattern(self):
        validator = SQLValidator()
        result = validator.validate("DROP TABLE users")
        assert not result["valid"]

    def test_allow_modifications_true(self):
        validator = SQLValidator(allow_modifications=True)
        result = validator.validate("SELECT 1")
        assert result["valid"]


class TestSQLQueryBuilder:
    def test_basic_select(self):
        qb = SQLQueryBuilder("users")
        sql = qb.select("id", "name").build()
        assert "SELECT id, name FROM users" in sql

    def test_select_all(self):
        qb = SQLQueryBuilder("items")
        sql = qb.build()
        assert "SELECT * FROM items" in sql

    def test_where_clause(self):
        qb = SQLQueryBuilder("products")
        sql = qb.select("*").where("price > 100").build()
        assert "WHERE" in sql
        assert "price > 100" in sql

    def test_order_by(self):
        qb = SQLQueryBuilder("tasks")
        sql = qb.select("*").order_by("created_at", desc=True).build()
        assert "ORDER BY" in sql
        assert "DESC" in sql

    def test_limit(self):
        qb = SQLQueryBuilder("logs")
        sql = qb.select("*").limit(10).build()
        assert "LIMIT 10" in sql

    def test_complex_query(self):
        qb = SQLQueryBuilder("orders")
        sql = (qb.select("id", "total")
               .where("total > 0")
               .order_by("total", desc=True)
               .limit(5)
               .build())
        assert "SELECT id, total FROM orders" in sql
        assert "WHERE total > 0" in sql
        assert "ORDER BY total DESC" in sql
        assert "LIMIT 5" in sql
