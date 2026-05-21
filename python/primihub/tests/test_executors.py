import pytest
import pandas as pd
from python.primihub.local.data_cleaning.executor import DataCleaningExecutor
from python.primihub.local.data_scaling.executor import DataScalingExecutor
from python.primihub.local.feature_binning.executor import FeatureBinningExecutor


class TestDataCleaningExecutor:
    def test_parse_params_defaults(self):
        executor = DataCleaningExecutor(
            common_params={"dataset": pd.DataFrame({"a": [1, None, 3]})}
        )
        assert executor.operations == ["missing", "duplicate"]

    def test_parse_params_custom(self):
        executor = DataCleaningExecutor(
            common_params={
                "operations": ["missing", "outlier"],
                "dataset": pd.DataFrame({"a": [1, 2, 3]}),
            }
        )
        assert "missing" in executor.operations
        assert "outlier" in executor.operations


class TestDataScalingExecutor:
    def test_parse_params(self):
        executor = DataScalingExecutor(
            common_params={
                "method": "standard",
                "dataset": pd.DataFrame({"a": [1, 2, 3]}),
            }
        )
        assert executor.method == "standard"


class TestFeatureBinningExecutor:
    def test_parse_params(self):
        executor = FeatureBinningExecutor(
            common_params={
                "method": "equal_width",
                "n_bins": 5,
                "dataset": pd.DataFrame({"a": range(100)}),
            }
        )
        assert executor.n_bins == 5
        assert executor.method == "equal_width"
