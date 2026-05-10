import pytest
import pandas as pd
import numpy as np
from python.primihub.local.feature_derivation.base import (
    PolynomialDeriver,
    InteractionDeriver,
    AggregationDeriver,
    DateTimeDeriver,
    MathDeriver,
)


class TestPolynomialDeriver:
    def test_basic_polynomial(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        deriver = PolynomialDeriver(degree=2, columns=["a"])
        result = deriver.derive(data)
        assert "a" in result.columns

    def test_no_poly_for_string(self):
        data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        deriver = PolynomialDeriver(degree=2)
        result = deriver.derive(data)
        assert "a" in result.columns
        assert "b" in result.columns


class TestInteractionDeriver:
    def test_basic_interaction(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        deriver = InteractionDeriver()
        result = deriver.derive(data)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_specific_columns(self):
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        deriver = InteractionDeriver(columns=["x", "y"])
        result = deriver.derive(data)
        assert "x" in result.columns
        assert "z" in result.columns


class TestDateTimeDeriver:
    def test_extract_features(self):
        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15", "2024-06-20", "2024-12-25"])
        })
        deriver = DateTimeDeriver(columns=["date"])
        result = deriver.derive(data)
        assert "date" in result.columns

    def test_no_datetime_columns(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        deriver = DateTimeDeriver()
        result = deriver.derive(data)
        assert "a" in result.columns


class TestAggregationDeriver:
    def test_basic_aggregation(self):
        data = pd.DataFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
        deriver = AggregationDeriver(group_columns=["cat"], agg_columns=["val"])
        result = deriver.derive(data)
        assert "cat" in result.columns
        assert "val" in result.columns


class TestMathDeriver:
    def test_basic_math(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        deriver = MathDeriver()
        result = deriver.derive(data)
        assert "a" in result.columns
        assert "b" in result.columns
