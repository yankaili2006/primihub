import pytest
import pandas as pd
import numpy as np
from python.primihub.local.data_cleaning.base import (
    MissingValueHandler,
    OutlierHandler,
    DuplicateHandler,
)


class TestMissingValueHandler:
    def test_drop_strategy(self):
        data = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        handler = MissingValueHandler(strategy="drop")
        result = handler.clean(data)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 1

    def test_drop_all_missing(self):
        data = pd.DataFrame({"a": [None, None], "b": [1, 2]})
        handler = MissingValueHandler(strategy="drop")
        result = handler.clean(data)
        assert len(result) == 0

    def test_fill_constant(self):
        data = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, 6]})
        handler = MissingValueHandler(strategy="fill", fill_value=0)
        result = handler.clean(data)
        assert result["a"].iloc[1] == 0
        assert result["b"].iloc[0] == 0

    def test_fill_mean(self):
        data = pd.DataFrame({"a": [1.0, None, 3.0]})
        handler = MissingValueHandler(strategy="fill", fill_value="mean")
        result = handler.clean(data)
        expected_mean = (1.0 + 3.0) / 2
        assert result["a"].iloc[1] == pytest.approx(expected_mean)

    def test_fill_median(self):
        data = pd.DataFrame({"a": [1.0, None, 10.0, 5.0]})
        handler = MissingValueHandler(strategy="fill", fill_value="median")
        result = handler.clean(data)
        expected_median = np.median([1.0, 10.0, 5.0])
        assert result["a"].iloc[1] == pytest.approx(expected_median)

    def test_fill_mode(self):
        data = pd.DataFrame({"a": [1, 1, 2, None]})
        handler = MissingValueHandler(strategy="fill", fill_value="mode")
        result = handler.clean(data)
        assert result["a"].iloc[3] == 1

    def test_fill_ffill(self):
        data = pd.DataFrame({"a": [1, None, None, 4]})
        handler = MissingValueHandler(strategy="fill", fill_value="ffill")
        result = handler.clean(data)
        assert result["a"].iloc[1] == 1
        assert result["a"].iloc[2] == 1

    def test_fill_bfill(self):
        data = pd.DataFrame({"a": [1, None, None, 4]})
        handler = MissingValueHandler(strategy="fill", fill_value="bfill")
        result = handler.clean(data)
        assert result["a"].iloc[1] == 4
        assert result["a"].iloc[2] == 4

    def test_specific_columns(self):
        data = pd.DataFrame({"a": [1, None, 3], "b": [None, "x", "y"]})
        handler = MissingValueHandler(strategy="fill", columns=["a"], fill_value=0)
        result = handler.clean(data)
        assert result["a"].iloc[1] == 0
        assert pd.isna(result["b"].iloc[0])

    def test_get_report(self):
        data = pd.DataFrame({"a": [1, None, 3]})
        handler = MissingValueHandler(strategy="fill", fill_value=0)
        handler.clean(data)
        report = handler.get_report()
        assert isinstance(report, dict)

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        handler = MissingValueHandler(strategy="drop")
        result = handler.clean(data)
        assert result.empty


class TestOutlierHandler:
    def test_zscore_drop(self):
        np.random.seed(42)
        data = pd.DataFrame({"a": [1, 2, 1, 2, 1, 2, 100, 2, 1, 2]})
        handler = OutlierHandler(method="zscore", threshold=2, strategy="drop")
        result = handler.clean(data)
        assert 100 not in result["a"].values

    def test_iqr_clip(self):
        data = pd.DataFrame({"a": list(range(20)) + [1000]})
        handler = OutlierHandler(method="iqr", strategy="clip")
        result = handler.clean(data)
        assert result["a"].max() < 1000

    def test_no_outliers(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        handler = OutlierHandler(method="zscore", threshold=3, strategy="drop")
        result = handler.clean(data)
        assert len(result) == len(data)

    def test_specific_columns(self):
        data = pd.DataFrame({"a": [1, 2, 1, 2, 100], "b": [1, 2, 3, 4, 5]})
        handler = OutlierHandler(method="zscore", threshold=2, strategy="drop", columns=["a"])
        result = handler.clean(data)
        assert len(result) >= 1

    def test_replace_mean(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0, 100.0]})
        handler = OutlierHandler(method="iqr", strategy="replace_mean")
        result = handler.clean(data)
        assert result["a"].max() < 100

    def test_get_report(self):
        data = pd.DataFrame({"a": [1, 2, 1, 2, 100]})
        handler = OutlierHandler(method="zscore", threshold=2, strategy="drop")
        handler.clean(data)
        report = handler.get_report()
        assert isinstance(report, dict)


class TestDuplicateHandler:
    def test_drop_all(self):
        data = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 3, 4, 4]})
        handler = DuplicateHandler(keep=False)
        result = handler.clean(data)
        assert len(result) == 0

    def test_keep_first(self):
        data = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        handler = DuplicateHandler(keep="first")
        result = handler.clean(data)
        assert len(result) == 2

    def test_keep_last(self):
        data = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 30]})
        handler = DuplicateHandler(keep="last")
        result = handler.clean(data)
        assert len(result) == 2

    def test_no_duplicates(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        handler = DuplicateHandler(keep="first")
        result = handler.clean(data)
        assert len(result) == 3

    def test_specific_subset(self):
        data = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        handler = DuplicateHandler(subset=["a"], keep="first")
        result = handler.clean(data)
        assert len(result) == 2

    def test_get_report(self):
        data = pd.DataFrame({"a": [1, 1, 2]})
        handler = DuplicateHandler(keep="first")
        handler.clean(data)
        report = handler.get_report()
        assert isinstance(report, dict)
