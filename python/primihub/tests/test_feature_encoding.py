import pytest
import pandas as pd
import numpy as np
from python.primihub.local.feature_encoding.base import (
    OneHotEncoder,
    LabelEncoder,
    FrequencyEncoder,
)


class TestOneHotEncoder:
    def test_basic_encoding(self):
        data = pd.DataFrame({"color": ["red", "blue", "green", "blue"]})
        encoder = OneHotEncoder()
        result = encoder.fit_transform(data)
        assert "color_blue" in result.columns
        assert "color_red" in result.columns
        assert "color_green" in result.columns

    def test_unknown_category(self):
        data = pd.DataFrame({"color": ["red", "blue"]})
        encoder = OneHotEncoder()
        encoder.fit(data)
        new_data = pd.DataFrame({"color": ["red", "green"]})
        result = encoder.transform(new_data)
        assert "color_red" in result.columns


class TestLabelEncoder:
    def test_basic_encoding(self):
        data = pd.DataFrame({"color": ["red", "blue", "green", "blue"]})
        encoder = LabelEncoder()
        result = encoder.fit_transform(data)
        assert result["color"].dtype in [np.int64, np.int32, np.int8]

    def test_three_categories(self):
        data = pd.DataFrame({"color": ["red", "blue", "green"]})
        encoder = LabelEncoder()
        result = encoder.fit_transform(data)
        assert result["color"].nunique() == 3


class TestFrequencyEncoder:
    def test_basic_encoding(self):
        data = pd.DataFrame({"color": ["red", "red", "blue", "blue", "blue"]})
        encoder = FrequencyEncoder()
        result = encoder.fit_transform(data)
        assert result["color"].iloc[0] < result["color"].iloc[2]
