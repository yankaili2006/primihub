import pytest
import pandas as pd
import numpy as np
from python.primihub.local.feature_selection.base import (
    VarianceSelector,
    CorrelationSelector,
)


class TestVarianceSelector:
    def test_low_variance_removed(self):
        data = pd.DataFrame({
            "a": [1, 1, 1, 1, 1],
            "b": [1, 2, 3, 4, 5],
            "c": [0, 0, 0, 0, 0],
        })
        selector = VarianceSelector(threshold=0.1)
        result = selector.fit_transform(data)
        assert "a" not in result.columns
        assert "b" in result.columns
        assert "c" not in result.columns

    def test_all_high_variance(self):
        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        selector = VarianceSelector(threshold=0.5)
        result = selector.fit_transform(data)
        assert len(result.columns) == 2

    def test_zero_threshold(self):
        data = pd.DataFrame({"a": [1, 1, 2, 2], "b": [0, 0, 0, 0]})
        selector = VarianceSelector(threshold=0.0)
        result = selector.fit_transform(data)
        assert "b" not in result.columns

    def test_get_selected_features(self):
        data = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        selector = VarianceSelector(threshold=0.5)
        selector.fit(data)
        selected = selector.get_selected_features()
        assert "b" in selected
        assert "a" not in selected

    def test_get_feature_scores(self):
        data = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        selector = VarianceSelector()
        selector.fit(data)
        scores = selector.get_feature_scores()
        assert "a" in scores
        assert "b" in scores
        assert scores["a"] == 0.0


class TestCorrelationSelector:
    def test_target_method_selects_high_correlation(self):
        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [100, 200, 300, 400, 500],
        })
        y = pd.Series([1, 2, 3, 4, 5])
        selector = CorrelationSelector(threshold=0.9, method="target",
                                        correlation_method="pearson")
        result = selector.fit_transform(data, y)
        assert len(result.columns) >= 1

    def test_fit_not_called(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        selector = CorrelationSelector()
        with pytest.raises(RuntimeError, match="not fitted"):
            selector.transform(data)

    def test_get_selected_features(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([1, 2, 3, 4, 5])
        selector = CorrelationSelector(threshold=0.9, method="target",
                                        correlation_method="pearson")
        selector.fit(data, y)
        selected = selector.get_selected_features()
        assert "a" in selected
