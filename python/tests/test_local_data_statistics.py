import numpy as np
import pandas as pd
import pytest

from primihub.local.data_statistics.base import (
    DescriptiveStatistics,
    DistributionAnalysis,
    CorrelationAnalysis,
    OutlierStatistics,
    MissingValueStatistics,
)


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "A": np.random.randn(100),
        "B": np.random.randn(100) * 5 + 10,
        "C": np.random.uniform(0, 100, 100),
        "D": np.random.poisson(5, 100).astype(float),
    })


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "a", "b", "c"],
        "int_col": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 5.0],
        "C": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def single_value_df():
    return pd.DataFrame({
        "X": [5.0, 5.0, 5.0, 5.0, 5.0],
        "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


class TestDescriptiveStatistics:
    def test_mean(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert abs(result["summary"][col]["mean"] - numeric_df[col].mean()) < 1e-10

    def test_variance(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert abs(result["summary"][col]["variance"] - numeric_df[col].var()) < 1e-10

    def test_std(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert abs(result["summary"][col]["std"] - numeric_df[col].std()) < 1e-10

    def test_min_max(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert result["summary"][col]["min"] == numeric_df[col].min()
            assert result["summary"][col]["max"] == numeric_df[col].max()
            assert result["summary"][col]["range"] == numeric_df[col].max() - numeric_df[col].min()

    def test_median_and_percentiles(self, numeric_df):
        ds = DescriptiveStatistics(percentiles=[0.25, 0.5, 0.75])
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            s = result["summary"][col]
            assert abs(s["median"] - numeric_df[col].median()) < 1e-10
            assert abs(s["percentiles"]["p25"] - numeric_df[col].quantile(0.25)) < 1e-10
            assert abs(s["percentiles"]["p50"] - numeric_df[col].quantile(0.5)) < 1e-10
            assert abs(s["percentiles"]["p75"] - numeric_df[col].quantile(0.75)) < 1e-10

    def test_count_and_missing(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert result["summary"][col]["count"] == len(numeric_df)

    def test_skewness_kurtosis(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert "skewness" in result["summary"][col]
            assert "kurtosis" in result["summary"][col]

    def test_selected_columns(self, numeric_df):
        ds = DescriptiveStatistics(columns=["A", "B"])
        result = ds.compute(numeric_df)
        assert set(result["summary"].keys()) == {"A", "B"}

    def test_mixed_df_numeric_only(self, mixed_df):
        ds = DescriptiveStatistics()
        result = ds.compute(mixed_df)
        assert "cat" not in result["summary"]
        assert "num1" in result["summary"]
        assert "num2" in result["summary"]

    def test_shape_and_dtypes(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        assert result["shape"]["rows"] == 100
        assert result["shape"]["columns"] == 4
        assert all(isinstance(v, str) for v in result["dtypes"].values())

    def test_single_value(self, single_value_df):
        ds = DescriptiveStatistics()
        result = ds.compute(single_value_df)
        assert result["summary"]["X"]["variance"] == 0.0
        assert result["summary"]["X"]["std"] == 0.0

    def test_empty_dataframe(self, empty_df):
        ds = DescriptiveStatistics()
        result = ds.compute(empty_df)
        assert result == {}

    def test_numeric_df_unique_count(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            assert result["summary"][col]["unique_count"] == len(numeric_df[col].unique())

    def test_mode(self, numeric_df):
        ds = DescriptiveStatistics()
        result = ds.compute(numeric_df)
        for col in numeric_df.columns:
            mode_val = numeric_df[col].mode().iloc[0]
            assert abs(result["summary"][col]["mode"] - mode_val) < 1e-10


class TestDistributionAnalysis:
    def test_histogram_shape(self, numeric_df):
        da = DistributionAnalysis(n_bins=10)
        result = da.compute(numeric_df)
        for col in numeric_df.columns:
            hist = result["distributions"][col]["histogram"]
            assert len(hist["counts"]) == 10
            assert len(hist["bin_edges"]) == 11
            assert len(hist["bin_centers"]) == 10

    def test_histogram_total_count(self, numeric_df):
        da = DistributionAnalysis(n_bins=5)
        result = da.compute(numeric_df)
        for col in numeric_df.columns:
            total = sum(result["distributions"][col]["histogram"]["counts"])
            assert total == len(numeric_df[col].dropna())

    def test_normality_test(self, numeric_df):
        da = DistributionAnalysis(test_normality=True)
        result = da.compute(numeric_df)
        for col in numeric_df.columns:
            nt = result["distributions"][col].get("normality_test")
            if nt:
                assert "shapiro_wilk" in nt
                assert "ks_test" in nt
                assert "statistic" in nt["shapiro_wilk"]
                assert "p_value" in nt["shapiro_wilk"]
                assert "is_normal" in nt["shapiro_wilk"]

    def test_selected_columns(self, numeric_df):
        da = DistributionAnalysis(columns=["A", "C"])
        result = da.compute(numeric_df)
        assert set(result["distributions"].keys()) == {"A", "C"}

    def test_too_few_samples(self):
        df = pd.DataFrame({"X": [1.0, 2.0]})
        da = DistributionAnalysis()
        result = da.compute(df)
        assert len(result["distributions"]) == 0

    def test_custom_bins(self, numeric_df):
        da = DistributionAnalysis(n_bins=20)
        result = da.compute(numeric_df)
        hist = result["distributions"]["A"]["histogram"]
        assert len(hist["counts"]) == 20

    def test_bin_centers(self, numeric_df):
        da = DistributionAnalysis(n_bins=4)
        result = da.compute(numeric_df)
        for col in numeric_df.columns:
            hist = result["distributions"][col]["histogram"]
            centers = hist["bin_centers"]
            edges = hist["bin_edges"]
            expected = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            assert np.allclose(centers, expected)


class TestCorrelationAnalysis:
    def test_correlation_matrix_shape(self, numeric_df):
        ca = CorrelationAnalysis()
        result = ca.compute(numeric_df)
        matrix = result["correlation_matrix"]
        cols = numeric_df.select_dtypes(include=[np.number]).columns
        assert set(matrix.keys()) == set(cols)

    def test_pearson_method(self, numeric_df):
        ca = CorrelationAnalysis(method="pearson")
        result = ca.compute(numeric_df)
        matrix = result["correlation_matrix"]
        cols = list(matrix.keys())
        pearson = numeric_df[cols].corr(method="pearson")
        for c1 in cols:
            for c2 in cols:
                assert abs(matrix[c1][c2] - pearson.loc[c1, c2]) < 1e-10

    def test_spearman_method(self, numeric_df):
        ca = CorrelationAnalysis(method="spearman")
        result = ca.compute(numeric_df)
        assert result["method"] == "spearman"

    def test_high_correlations(self):
        np.random.seed(42)
        A = np.random.randn(100)
        df = pd.DataFrame({
            "A": A,
            "B": A * 2 + np.random.randn(100) * 0.1,
            "C": np.random.randn(100),
        })
        ca = CorrelationAnalysis(threshold=0.5)
        result = ca.compute(df)
        assert len(result["high_correlations"]) > 0
        pair = result["high_correlations"][0]
        assert pair["feature1"] == "A"
        assert pair["feature2"] == "B"

    def test_selected_columns(self, numeric_df):
        ca = CorrelationAnalysis(columns=["A", "B", "C"])
        result = ca.compute(numeric_df)
        assert set(result["correlation_matrix"].keys()) == {"A", "B", "C"}

    def test_single_column_returns_empty(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        ca = CorrelationAnalysis()
        result = ca.compute(df)
        assert result == {}

    def test_high_correlations_sorted(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
        })
        ca = CorrelationAnalysis(threshold=0.5)
        result = ca.compute(df)
        corrs = [abs(p["correlation"]) for p in result["high_correlations"]]
        assert corrs == sorted(corrs, reverse=True)


class TestOutlierStatistics:
    def test_iqr_method(self, numeric_df):
        os = OutlierStatistics(method="iqr", threshold=1.5)
        result = os.compute(numeric_df)
        for col in numeric_df.columns:
            stats = result["columns"][col]
            assert "outlier_count" in stats
            assert "outlier_rate" in stats
            assert 0 <= stats["outlier_rate"] <= 1
            assert stats["lower_bound"] <= stats["upper_bound"]

    def test_zscore_method(self, numeric_df):
        os = OutlierStatistics(method="zscore", threshold=3.0)
        result = os.compute(numeric_df)
        for col in numeric_df.columns:
            stats = result["columns"][col]
            assert "outlier_count" in stats
            assert "outlier_rate" in stats

    def test_outlier_indices(self, numeric_df):
        os = OutlierStatistics(method="iqr")
        result = os.compute(numeric_df)
        for col in numeric_df.columns:
            indices = result["columns"][col]["outlier_indices"]
            assert len(indices) <= 100

    def test_iqr_bounds_consistency(self):
        df = pd.DataFrame({"X": [1, 2, 3, 4, 5, 100]})
        os = OutlierStatistics(method="iqr", threshold=1.5)
        result = os.compute(df)
        assert result["columns"]["X"]["outlier_count"] > 0

    def test_zscore_bounds_consistency(self):
        df = pd.DataFrame({"X": [1, 2, 3, 4, 5, 100]})
        os = OutlierStatistics(method="zscore", threshold=2.0)
        result = os.compute(df)
        assert result["columns"]["X"]["outlier_count"] > 0

    def test_no_outliers(self):
        df = pd.DataFrame({"X": [5, 5, 5, 5, 5]})
        os = OutlierStatistics(method="iqr")
        result = os.compute(df)
        assert result["columns"]["X"]["outlier_count"] == 0

    def test_selected_columns(self, numeric_df):
        os = OutlierStatistics(columns=["A", "B"])
        result = os.compute(numeric_df)
        assert set(result["columns"].keys()) == {"A", "B"}

    def test_method_metadata(self, numeric_df):
        os = OutlierStatistics(method="iqr", threshold=2.0)
        result = os.compute(numeric_df)
        assert result["method"] == "iqr"
        assert result["threshold"] == 2.0


class TestMissingValueStatistics:
    def test_no_missing(self, numeric_df):
        mv = MissingValueStatistics()
        result = mv.compute(numeric_df)
        assert result["total_missing"] == 0
        assert result["overall_missing_rate"] == 0.0
        for col in numeric_df.columns:
            assert result["columns"][col]["missing_count"] == 0

    def test_with_missing(self, df_with_missing):
        mv = MissingValueStatistics()
        result = mv.compute(df_with_missing)
        assert result["total_missing"] == 3
        assert result["columns"]["A"]["missing_count"] == 1
        assert result["columns"]["B"]["missing_count"] == 2
        assert result["columns"]["C"]["missing_count"] == 0

    def test_overall_missing_rate(self, df_with_missing):
        mv = MissingValueStatistics()
        result = mv.compute(df_with_missing)
        expected = 3 / (5 * 3)
        assert abs(result["overall_missing_rate"] - expected) < 1e-10

    def test_rows_with_missing(self, df_with_missing):
        mv = MissingValueStatistics()
        result = mv.compute(df_with_missing)
        assert result["rows_with_missing"] == 3
        assert result["complete_rows"] == 2

    def test_all_missing(self):
        df = pd.DataFrame({"A": [np.nan, np.nan], "B": [1, 2]})
        mv = MissingValueStatistics()
        result = mv.compute(df)
        assert result["total_missing"] == 2
        assert result["columns"]["A"]["missing_rate"] == 1.0
        assert result["columns"]["B"]["missing_rate"] == 0.0

    def test_selected_columns(self, df_with_missing):
        mv = MissingValueStatistics(columns=["A"])
        result = mv.compute(df_with_missing)
        assert set(result["columns"].keys()) == {"A"}
        assert result["total_cells"] == 5

    def test_columns_by_missing_rate(self, df_with_missing):
        mv = MissingValueStatistics()
        result = mv.compute(df_with_missing)
        sorted_cols = result["columns_by_missing_rate"]
        rates = [r[1]["missing_rate"] for r in sorted_cols]
        assert rates == sorted(rates, reverse=True)

    def test_non_missing_count(self, df_with_missing):
        mv = MissingValueStatistics()
        result = mv.compute(df_with_missing)
        assert result["columns"]["A"]["non_missing_count"] == 4
        assert result["columns"]["B"]["non_missing_count"] == 3
        assert result["columns"]["C"]["non_missing_count"] == 5


class TestEdgeCases:
    def test_all_nan_column(self):
        df = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [1, 2, 3]})
        ds = DescriptiveStatistics()
        result = ds.compute(df)
        assert "A" not in result["summary"]
        assert "B" in result["summary"]

    def test_inf_values(self):
        df = pd.DataFrame({"A": [1.0, np.inf, 3.0], "B": [4.0, 5.0, 6.0]})
        ds = DescriptiveStatistics()
        result = ds.compute(df)
        assert "A" in result["summary"]

    def test_string_column_ignored(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        ds = DescriptiveStatistics()
        result = ds.compute(df)
        assert "A" in result["summary"]
        assert "B" not in result["summary"]

    def test_mixed_types_dataframe(self, mixed_df):
        ds = DescriptiveStatistics()
        result = ds.compute(mixed_df)
        assert "num1" in result["summary"]
        assert "num2" in result["summary"]
        assert "int_col" in result["summary"]
        assert "cat" not in result["summary"]

    def test_large_range(self):
        df = pd.DataFrame({"A": [1e-10, 1e10]})
        ds = DescriptiveStatistics()
        result = ds.compute(df)
        assert abs(result["summary"]["A"]["range"] - 1e10) < 1e-5
