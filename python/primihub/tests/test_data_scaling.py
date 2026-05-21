import pytest
import pandas as pd
import numpy as np
from python.primihub.local.data_scaling.base import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)


class TestStandardScaler:
    def test_basic_scaling(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        scaler = StandardScaler()
        result = scaler.fit_transform(data)
        assert abs(result["a"].mean()) < 0.01

    def test_specific_columns(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        scaler = StandardScaler(columns=["a"])
        result = scaler.fit_transform(data)
        assert abs(result["a"].mean()) < 0.01
        assert result["b"].iloc[0] == 10

    def test_constant_column(self):
        data = pd.DataFrame({"a": [5, 5, 5]})
        scaler = StandardScaler()
        result = scaler.fit_transform(data)
        assert result["a"].isnull().sum() == 0


class TestMinMaxScaler:
    def test_basic_scaling(self):
        data = pd.DataFrame({"a": [0, 50, 100]})
        scaler = MinMaxScaler()
        result = scaler.fit_transform(data)
        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0

    def test_custom_range(self):
        data = pd.DataFrame({"a": [0, 50, 100]})
        scaler = MinMaxScaler(feature_range=(-1, 1))
        result = scaler.fit_transform(data)
        assert result["a"].min() == -1.0
        assert result["a"].max() == 1.0

    def test_single_value(self):
        data = pd.DataFrame({"a": [42, 42, 42]})
        scaler = MinMaxScaler()
        result = scaler.fit_transform(data)
        assert result["a"].iloc[0] == 0.0


class TestRobustScaler:
    def test_basic_scaling(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        scaler = RobustScaler()
        result = scaler.fit_transform(data)
        assert abs(result["a"].median()) < 0.001


class TestMaxAbsScaler:
    def test_basic_scaling(self):
        data = pd.DataFrame({"a": [-2, -1, 0, 1, 2]})
        scaler = MaxAbsScaler()
        result = scaler.fit_transform(data)
        assert result["a"].max() == 1.0
        assert result["a"].min() == -1.0

    def test_all_positive(self):
        data = pd.DataFrame({"a": [10, 20, 30]})
        scaler = MaxAbsScaler()
        result = scaler.fit_transform(data)
        assert result["a"].max() == 1.0
        assert result["a"].min() == 1.0/3


class TestNormalizer:
    def test_l1_normalization(self):
        data = pd.DataFrame({"a": [3, 0], "b": [4, 5]})
        scaler = Normalizer(norm="l1")
        result = scaler.fit_transform(data)
        row0_sum = abs(result.iloc[0]).sum()
        assert abs(row0_sum - 1.0) < 0.001

    def test_l2_normalization(self):
        data = pd.DataFrame({"a": [3, 0], "b": [4, 5]})
        scaler = Normalizer(norm="l2")
        result = scaler.fit_transform(data)
        row0_l2 = (result.iloc[0]**2).sum()
        assert abs(row0_l2 - 1.0) < 0.01
