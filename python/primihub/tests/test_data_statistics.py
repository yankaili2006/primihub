import pytest
import pandas as pd
import numpy as np
from python.primihub.local.data_statistics.base import (
    DescriptiveStatistics,
    CorrelationAnalysis,
)


class TestDescriptiveStatistics:
    def test_basic_stats(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        stat = DescriptiveStatistics()
        result = stat.compute(data)
        assert "summary" in result
        assert "shape" in result
        assert "dtypes" in result
        assert "a" in result["summary"]
        assert "b" in result["summary"]
        assert result["shape"]["rows"] == 5

    def test_custom_percentiles(self):
        data = pd.DataFrame({"a": range(1, 101)})
        stat = DescriptiveStatistics(percentiles=[0.1, 0.5, 0.9])
        result = stat.compute(data)
        summary = result["summary"]
        assert "a" in summary

    def test_string_columns_excluded(self):
        data = pd.DataFrame({
            "num": [1, 2, 3],
            "txt": ["a", "b", "c"],
        })
        stat = DescriptiveStatistics()
        result = stat.compute(data)
        summary = result["summary"]
        assert "num" in summary
        assert "txt" not in summary

    def test_single_column(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        stat = DescriptiveStatistics()
        result = stat.compute(data)
        assert "summary" in result
        assert "a" in result["summary"]


class TestCorrelationAnalysis:
    def test_pearson_correlation(self):
        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
        })
        corr = CorrelationAnalysis(method="pearson")
        result = corr.compute(data)
        assert "correlation_matrix" in result
        matrix = result["correlation_matrix"]
        assert matrix["a"]["b"] == pytest.approx(1.0, abs=0.01)
        assert matrix["a"]["c"] == pytest.approx(-1.0, abs=0.01)

    def test_spearman_correlation(self):
        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        corr = CorrelationAnalysis(method="spearman")
        result = corr.compute(data)
        matrix = result["correlation_matrix"]
        assert matrix["a"]["b"] == pytest.approx(-1.0, abs=0.01)

    def test_specific_columns(self):
        data = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })
        corr = CorrelationAnalysis(columns=["a", "b"])
        result = corr.compute(data)
        matrix = result["correlation_matrix"]
        assert "a" in matrix
        assert "b" in matrix
        assert "c" not in matrix

    def test_constant_column(self):
        data = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        corr = CorrelationAnalysis(method="pearson")
        result = corr.compute(data)
        assert "correlation_matrix" in result
