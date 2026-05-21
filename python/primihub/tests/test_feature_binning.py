import pytest
import pandas as pd
import numpy as np
from python.primihub.local.feature_binning.base import (
    EqualWidthBinner,
    EqualFrequencyBinner,
)


class TestEqualWidthBinner:
    def test_basic_binning(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        binner = EqualWidthBinner(n_bins=5)
        result = binner.fit_transform(data)
        assert result["a"].nunique() <= 5

    def test_specific_columns(self):
        data = pd.DataFrame({"a": range(20), "b": range(20, 40)})
        binner = EqualWidthBinner(n_bins=4, columns=["a"])
        result = binner.fit_transform(data)
        assert result["a"].nunique() <= 4
        assert result["b"].nunique() == 20

    def test_get_bin_edges(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        binner = EqualWidthBinner(n_bins=2)
        binner.fit(data)
        edges = binner.get_bin_edges()
        assert "a" in edges


class TestEqualFrequencyBinner:
    def test_basic_binning(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        binner = EqualFrequencyBinner(n_bins=5)
        result = binner.fit_transform(data)
        assert result["a"].nunique() <= 5

    def test_specific_columns(self):
        data = pd.DataFrame({"a": range(30), "b": range(30, 60)})
        binner = EqualFrequencyBinner(n_bins=3, columns=["b"])
        result = binner.fit_transform(data)
        assert result["b"].nunique() <= 3
        assert result["a"].nunique() == 30
