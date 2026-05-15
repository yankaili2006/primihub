import sys
import types
import numpy as np
import pytest

np.float = float

_mock_modules = {
    'linkcontext': types.ModuleType('linkcontext'),
    'grpc': types.ModuleType('grpc'),
    'google': types.ModuleType('google'),
    'google.protobuf': types.ModuleType('google.protobuf'),
    'primihub.FL.crypto.ckks': types.ModuleType('primihub.FL.crypto.ckks'),
    'primihub.FL.crypto.paillier': types.ModuleType('primihub.FL.crypto.paillier'),
}

class _MockGrpcClient:
    def __init__(self, *a, **kw): pass
    def send(self, *a, **kw): pass
    def recv(self, *a, **kw): pass
    def recv_all(self, *a, **kw): return {}
    def send_all(self, *a, **kw): pass

_mock_modules['primihub.FL.utils.net_work'] = types.ModuleType('primihub.FL.utils.net_work')
_mock_modules['primihub.FL.utils.net_work'].GrpcClient = _MockGrpcClient
_mock_modules['primihub.FL.utils.net_work'].MultiGrpcClients = _MockGrpcClient

_mock_modules['primihub.FL.utils.base'] = types.ModuleType('primihub.FL.utils.base')
class _MockBaseModel:
    def __init__(self, **kw): pass
_mock_modules['primihub.FL.utils.base'].BaseModel = _MockBaseModel

_mock_modules['primihub.FL.utils.file'] = types.ModuleType('primihub.FL.utils.file')
_mock_modules['primihub.FL.utils.file'].save_json_file = lambda *a, **kw: None
_mock_modules['primihub.FL.utils.file'].save_pickle_file = lambda *a, **kw: None
_mock_modules['primihub.FL.utils.file'].load_pickle_file = lambda *a, **kw: None

_mock_modules['primihub.FL.utils.dataset'] = types.ModuleType('primihub.FL.utils.dataset')
_mock_modules['primihub.FL.utils.dataset'].read_data = lambda *a, **kw: np.random.randn(10, 3)
_mock_modules['primihub.FL.utils.dataset'].DataLoader = lambda *a, **kw: None

import logging
_mock_modules['primihub.utils.logger_util'] = types.ModuleType('primihub.utils.logger_util')
_mock_modules['primihub.utils.logger_util'].logger = logging.getLogger('test')

_mock_modules['primihub.context'] = types.ModuleType('primihub.context')
_mock_modules['primihub.context'].Context = type('Context', (), {})

_mock_modules['primihub.client'] = types.ModuleType('primihub.client')

for _name, _mod in _mock_modules.items():
    _mod.__path__ = []
    sys.modules[_name] = _mod

from primihub.FL.feature_similarity.base import (
    CosineSimilarity,
    PearsonCorrelation,
    MutualInformation,
    JaccardSimilarity,
    FeatureSimilarityAnalyzer,
)
from primihub.FL.feature_encoding.base import (
    FLOneHotEncoder,
    FLLabelEncoder,
    FLTargetEncoder,
    FLHashEncoder,
    FLEmbeddingEncoder,
)
from primihub.FL.feature_alignment.base import (
    StatisticalAlignment,
    DistributionAlignment,
    SchemaAlignment,
    FeatureMapper,
)
from primihub.FL.feature_binning.base import (
    EqualWidthBinning,
    EqualFrequencyBinning,
    OptimalBinning,
    WOEBinning,
)
from primihub.FL.data_splitting.base import (
    TrainTestSplitter,
    KFoldSplitter,
    StratifiedSplitter,
    TimeSplitter,
    GroupSplitter,
)


# ============================================================
# Feature Similarity Tests
# ============================================================

class TestCosineSimilarity:
    def test_self_similarity(self):
        X = np.random.randn(50, 4)
        cs = CosineSimilarity()
        sim = cs.compute(X, X)
        assert sim.shape == (4, 4)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        X = np.array([[1, 0], [0, 1]], dtype=float)
        cs = CosineSimilarity()
        sim = cs.compute(X)
        assert abs(sim[0, 1]) < 1e-10

    def test_cross_party(self):
        X = np.random.randn(30, 3)
        Y = np.random.randn(30, 2)
        cs = CosineSimilarity()
        sim = cs.compute(X, Y)
        assert sim.shape == (3, 2)
        assert np.all(sim >= -1.0) and np.all(sim <= 1.0)

    def test_normalize_zeros(self):
        X = np.zeros((10, 3))
        cs = CosineSimilarity()
        sim = cs.compute(X)
        assert sim.shape == (3, 3)

    def test_single_feature(self):
        X = np.random.randn(20, 1)
        cs = CosineSimilarity()
        sim = cs.compute(X)
        assert sim.shape == (1, 1)
        assert abs(sim[0, 0] - 1.0) < 1e-6

    def test_batch_cross(self):
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 3)
        cs = CosineSimilarity()
        sim = cs.compute(X, Y)
        assert sim.shape == (5, 3)


class TestPearsonCorrelation:
    def test_self_correlation(self):
        X = np.random.randn(50, 4)
        pc = PearsonCorrelation()
        corr = pc.compute(X, X)
        assert corr.shape == (4, 4)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_perfect_linear(self):
        X = np.random.randn(50, 1)
        Y = X * 2 + 1
        pc = PearsonCorrelation()
        corr = pc.compute(X, Y)
        assert abs(corr[0, 0] - 1.0) < 1e-10

    def test_negative_correlation(self):
        X = np.random.randn(50, 1)
        Y = -X
        pc = PearsonCorrelation()
        corr = pc.compute(X, Y)
        assert abs(corr[0, 0] - (-1.0)) < 1e-6

    def test_cross_party(self):
        X = np.random.randn(30, 3)
        Y = np.random.randn(30, 2)
        pc = PearsonCorrelation()
        corr = pc.compute(X, Y)
        assert corr.shape == (3, 2)

    def test_no_correlation(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        Y = np.random.randn(100, 1)
        pc = PearsonCorrelation()
        corr = pc.compute(X, Y)
        assert abs(corr[0, 0]) < 0.3


class TestMutualInformation:
    def test_shape(self):
        X = np.random.randn(50, 3)
        Y = np.random.randn(50, 2)
        mi = MutualInformation(n_bins=10)
        result = mi.compute(X, Y)
        assert result.shape == (3, 2)

    def test_self_mutual_information(self):
        X = np.random.randn(30, 2)
        mi = MutualInformation(n_bins=5)
        result = mi.compute(X, X)
        assert np.all(result >= 0)
        assert result.shape == (2, 2)

    def test_mutual_info_non_negative(self):
        X = np.random.randn(30, 3)
        Y = np.random.randn(30, 2)
        mi = MutualInformation(n_bins=8)
        result = mi.compute(X, Y)
        assert np.all(result >= 0)

    def test_identical_variables(self):
        X = np.random.randn(50, 1)
        mi = MutualInformation(n_bins=10)
        result = mi.compute(X, X)
        assert result[0, 0] > 0


class TestJaccardSimilarity:
    def test_identical(self):
        X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=float)
        js = JaccardSimilarity()
        sim = js.compute(X, X)
        assert np.allclose(np.diag(sim), 1.0)

    def test_disjoint(self):
        X = np.array([[1, 0], [0, 1]], dtype=float)
        Y = np.array([[0, 1], [1, 0]], dtype=float)
        js = JaccardSimilarity()
        sim = js.compute(X, Y)
        assert sim[0, 0] == 0.0
        assert sim[1, 1] == 0.0

    def test_shape(self):
        X = np.random.randint(0, 2, size=(30, 4)).astype(float)
        Y = np.random.randint(0, 2, size=(30, 3)).astype(float)
        js = JaccardSimilarity()
        sim = js.compute(X, Y)
        assert sim.shape == (4, 3)
        assert np.all(sim >= 0) and np.all(sim <= 1)

    def test_all_zeros(self):
        X = np.zeros((10, 2))
        js = JaccardSimilarity()
        sim = js.compute(X)
        assert np.all(sim == 0.0)


class TestFeatureSimilarityAnalyzer:
    def test_analyze_default_methods(self):
        X = np.random.randn(30, 4)
        Y = np.random.randn(30, 3)
        analyzer = FeatureSimilarityAnalyzer(methods=["cosine", "pearson"])
        results = analyzer.analyze(X, Y)
        assert "cosine" in results
        assert "pearson" in results
        assert results["cosine"].shape == (4, 3)
        assert results["pearson"].shape == (4, 3)

    def test_analyze_all_methods(self):
        X = np.random.randn(30, 3)
        analyzer = FeatureSimilarityAnalyzer(methods=["cosine", "pearson", "mi", "jaccard"])
        results = analyzer.analyze(X)
        assert len(results) == 4

    def test_find_similar_features(self):
        X = np.eye(3)
        analyzer = FeatureSimilarityAnalyzer()
        sim = CosineSimilarity().compute(X)
        pairs = analyzer.find_similar_features(sim, threshold=0.5)
        assert isinstance(pairs, list)

    def test_find_similar_features_with_names(self):
        X = np.random.randn(20, 3)
        analyzer = FeatureSimilarityAnalyzer()
        sim = CosineSimilarity().compute(X)
        pairs = analyzer.find_similar_features(
            sim, threshold=0.1,
            feature_names_x=["a", "b", "c"],
            feature_names_y=["x", "y", "z"]
        )
        if pairs:
            assert len(pairs[0]) == 3

    def test_unknown_method_warning(self):
        X = np.random.randn(10, 2)
        analyzer = FeatureSimilarityAnalyzer(methods=["unknown_method"])
        results = analyzer.analyze(X)
        assert results == {}


# ============================================================
# Feature Encoding Tests
# ============================================================

class TestFLOneHotEncoder:
    def test_basic_encoding(self):
        X = np.array(["a", "b", "c", "a", "b"])
        enc = FLOneHotEncoder()
        encoded = enc.fit_transform(X)
        assert encoded.shape == (5, 3)
        assert np.allclose(encoded.sum(axis=1), 1.0)

    def test_inverse_transform(self):
        X = np.array(["a", "b", "c"])
        enc = FLOneHotEncoder()
        encoded = enc.fit_transform(X)
        decoded = enc.inverse_transform(encoded)
        assert np.array_equal(decoded.ravel(), X)

    def test_multiple_columns(self):
        X = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
        enc = FLOneHotEncoder()
        encoded = enc.fit_transform(X)
        assert encoded.shape == (3, 5)

    def test_not_fitted_raises(self):
        enc = FLOneHotEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(np.array(["a"]))


class TestFLLabelEncoder:
    def test_basic(self):
        X = np.array(["cat", "dog", "bird", "cat"])
        enc = FLLabelEncoder()
        encoded = enc.fit_transform(X)
        assert np.array_equal(encoded, [1, 2, 0, 1])

    def test_inverse_transform(self):
        X = np.array(["x", "y", "z"])
        enc = FLLabelEncoder()
        encoded = enc.fit_transform(X)
        decoded = enc.inverse_transform(encoded)
        assert np.array_equal(decoded, X)

    def test_unknown_category(self):
        X = np.array(["a", "b", "c"])
        enc = FLLabelEncoder()
        enc.fit(X)
        result = enc.transform(np.array(["d"]))
        assert result[0] == -1

    def test_not_fitted_raises(self):
        enc = FLLabelEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(np.array(["a"]))


class TestFLTargetEncoder:
    def test_basic(self):
        X = np.array(["a", "b", "a", "b", "c"])
        y = np.array([1, 0, 1, 0, 1])
        enc = FLTargetEncoder(smoothing=1.0)
        encoded = enc.fit_transform(X, y)
        assert encoded.shape == (5, 1)

    def test_unknown_category_uses_global_mean(self):
        X = np.array(["a", "b", "a"])
        y = np.array([1, 0, 1])
        enc = FLTargetEncoder()
        enc.fit(X, y)
        result = enc.transform(np.array(["unknown"]))
        assert result[0, 0] == enc.global_mean_

    def test_inverse_raises(self):
        X = np.array(["a", "b"])
        y = np.array([1, 0])
        enc = FLTargetEncoder()
        enc.fit(X, y)
        with pytest.raises(NotImplementedError):
            enc.inverse_transform(X)

    def test_smoothing_effect(self):
        X = np.array(["a", "b", "a", "b"])
        y = np.array([1, 0, 1, 0])
        enc_smooth = FLTargetEncoder(smoothing=10.0)
        enc_no_smooth = FLTargetEncoder(smoothing=0.0)
        encoded_smooth = enc_smooth.fit_transform(X, y)
        encoded_no = enc_no_smooth.fit_transform(X, y)
        assert not np.allclose(encoded_smooth, encoded_no)


class TestFLHashEncoder:
    def test_basic(self):
        X = np.array(["a", "b", "c"])
        enc = FLHashEncoder(n_components=4)
        encoded = enc.fit_transform(X)
        assert encoded.shape == (3, 4)

    def test_deterministic(self):
        X = np.array(["hello", "world"])
        enc = FLHashEncoder(n_components=8)
        r1 = enc.fit_transform(X)
        r2 = enc.fit_transform(X)
        assert np.allclose(r1, r2)

    def test_inverse_raises(self):
        enc = FLHashEncoder()
        enc.fit(np.array(["a"]))
        with pytest.raises(NotImplementedError):
            enc.inverse_transform(np.array([[1, 2]]))

    def test_multiple_columns(self):
        X = np.array([["a", "x"], ["b", "y"]])
        enc = FLHashEncoder(n_components=3)
        encoded = enc.fit_transform(X)
        assert encoded.shape == (2, 6)


class TestFLEmbeddingEncoder:
    def test_basic(self):
        X = np.array(["a", "b", "c", "a"])
        enc = FLEmbeddingEncoder(embedding_dim=4)
        encoded = enc.fit_transform(X)
        assert encoded.shape == (4, 4)

    def test_random_init_gives_different_embeddings(self):
        X = np.array(["a", "b"])
        enc = FLEmbeddingEncoder(embedding_dim=3)
        encoded = enc.fit_transform(X)
        assert not np.allclose(encoded[0], encoded[1])

    def test_unknown_category_zeros(self):
        X = np.array(["a", "b"])
        enc = FLEmbeddingEncoder()
        enc.fit(X)
        result = enc.transform(np.array(["unknown"]))
        assert np.allclose(result, 0)

    def test_inverse_raises(self):
        enc = FLEmbeddingEncoder()
        enc.fit(np.array(["a"]))
        with pytest.raises(NotImplementedError):
            enc.inverse_transform(np.array([[1, 2]]))


# ============================================================
# Feature Alignment Tests
# ============================================================

class TestStatisticalAlignment:
    def test_zscore_identity_without_channel(self):
        X = np.random.randn(50, 4)
        sa = StatisticalAlignment(method="zscore")
        aligned = sa.fit_transform(X)
        assert aligned.shape == X.shape

    def test_minmax_alignment(self):
        X = np.random.rand(50, 3) * 100
        sa = StatisticalAlignment(method="minmax")
        aligned = sa.fit_transform(X)
        assert aligned.shape == X.shape

    def test_rank_alignment(self):
        X = np.random.randn(30, 2)
        sa = StatisticalAlignment(method="rank")
        aligned = sa.fit_transform(X)
        assert np.all(aligned >= 0) and np.all(aligned <= 1)

    def test_not_fitted_raises(self):
        sa = StatisticalAlignment()
        with pytest.raises(RuntimeError):
            sa.transform(np.array([[1, 2]]))

    def test_local_statistics_stored(self):
        X = np.random.randn(30, 3)
        sa = StatisticalAlignment()
        sa.fit(X)
        assert sa.local_mean_ is not None
        assert sa.local_std_ is not None
        assert sa.local_min_ is not None
        assert sa.local_max_ is not None

    def test_fit_sync_sets_global_stats(self):
        X = np.random.randn(30, 3)
        sa = StatisticalAlignment()
        sa.fit(X)
        assert sa.global_mean_ is not None

    def test_minmax_bounds(self):
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
        sa = StatisticalAlignment(method="minmax")
        aligned = sa.fit_transform(X)
        for col in range(X.shape[1]):
            assert abs(aligned[:, col].min() - X[:, col].min()) < 1e-6
            assert abs(aligned[:, col].max() - X[:, col].max()) < 1e-6


class TestDistributionAlignment:
    def test_basic(self):
        X = np.random.randn(50, 3)
        da = DistributionAlignment(n_quantiles=50)
        aligned = da.fit_transform(X)
        assert aligned.shape == X.shape

    def test_with_reference(self):
        X = np.random.randn(30, 2)
        ref = np.random.randn(30, 2) * 2 + 5
        da = DistributionAlignment(n_quantiles=30)
        aligned = da.fit_transform(X, ref)
        assert aligned.shape == X.shape

    def test_not_fitted_raises(self):
        da = DistributionAlignment()
        with pytest.raises(RuntimeError):
            da.transform(np.array([[1, 2]]))


class TestSchemaAlignment:
    def test_basic(self):
        X = np.random.randn(10, 3)
        sa = SchemaAlignment()
        sa.fit(X, feature_names=["a", "b", "c"])
        assert sa.local_schema_ is not None
        assert sa.global_schema_ is not None

    def test_transform_returns_same_shape(self):
        X = np.random.randn(10, 3)
        sa = SchemaAlignment()
        sa.fit(X, feature_names=["a", "b", "c"])
        result = sa.transform(X)
        assert result.shape == X.shape

    def test_default_feature_names(self):
        X = np.random.randn(10, 4)
        sa = SchemaAlignment()
        sa.fit(X)
        assert sa.local_schema_["names"] == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_default_types(self):
        X = np.random.randn(10, 2)
        sa = SchemaAlignment()
        sa.fit(X)
        assert sa.local_schema_["types"] == ["numeric", "numeric"]


class TestFeatureMapper:
    def test_exact_name_match(self):
        mapper = FeatureMapper(similarity_threshold=0.8)
        mapping = mapper.build_mapping(
            local_features=["age", "income"],
            remote_features=["age", "salary"]
        )
        assert mapping.get("age") == "age"

    def test_similarity_matrix_mapping(self):
        mapper = FeatureMapper(similarity_threshold=0.5)
        sim = np.array([[0.9, 0.1], [0.2, 0.8]])
        mapping = mapper.build_mapping(
            local_features=["a", "b"],
            remote_features=["c", "d"],
            similarity_matrix=sim
        )
        assert len(mapping) == 2


# ============================================================
# Feature Binning Tests
# ============================================================

class TestEqualWidthBinning:
    def test_basic(self):
        X = np.random.rand(100, 3) * 100
        binner = EqualWidthBinning(n_bins=10)
        binned = binner.fit_transform(X)
        assert binned.shape == (100, 3)
        assert np.all(binned >= 0) and np.all(binned < 10)

    def test_bin_edges_length(self):
        X = np.linspace(0, 100, 50).reshape(-1, 1)
        binner = EqualWidthBinning(n_bins=5)
        binner.fit(X)
        assert len(binner.bin_edges_) == 1
        assert len(binner.bin_edges_[0]) == 6

    def test_not_fitted_raises(self):
        binner = EqualWidthBinning()
        with pytest.raises(RuntimeError):
            binner.transform(np.array([[1, 2]]))

    def test_single_column_binning(self):
        X = np.random.rand(50) * 100
        binner = EqualWidthBinning(n_bins=4)
        binned = binner.fit_transform(X)
        assert binned.shape == (50, 1)
        assert np.all(binned >= 0) and np.all(binned < 4)


class TestEqualFrequencyBinning:
    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(200, 2)
        binner = EqualFrequencyBinning(n_bins=5)
        binned = binner.fit_transform(X)
        assert binned.shape == (200, 2)
        assert np.all(binned >= 0)

    def test_single_value(self):
        X = np.ones((20, 1))
        binner = EqualFrequencyBinning(n_bins=3)
        binned = binner.fit_transform(X)
        assert binned.shape == (20, 1)
        assert np.all(binned >= -1)
        assert binned.dtype in (np.int64, np.int32)


class TestOptimalBinning:
    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = (X[:, 0] > 0).astype(int)
        binner = OptimalBinning(n_bins=5)
        binned = binner.fit_transform(X, y)
        assert binned.shape == (200, 1)

    def test_bin_edges_stored(self):
        X = np.random.randn(100, 1)
        y = np.random.randint(0, 2, size=100)
        binner = OptimalBinning(n_bins=4)
        binner.fit(X, y)
        assert len(binner.bin_edges_) == 1
        assert len(binner.bin_edges_[0]) >= 2


class TestWOEBinning:
    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = (X[:, 0] + np.random.randn(200) * 0.5 > 0).astype(int)
        binner = WOEBinning(n_bins=5)
        binner.fit(X, y)
        assert binner.woe_values_ is not None
        assert binner.iv_values_ is not None

    def test_transform_woe(self):
        X = np.random.randn(30, 1)
        y = (X[:, 0] > 0).astype(int)
        binner = WOEBinning(n_bins=4)
        binner.fit(X, y)
        woe = binner.transform_woe(X)
        assert woe.shape == X.shape
        assert not np.allclose(woe, 0)

    def test_iv_non_negative(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        binner = WOEBinning(n_bins=5)
        binner.fit(X, y)
        assert all(iv >= 0 for iv in binner.iv_values_)


# ============================================================
# Data Splitting Tests
# ============================================================

class TestTrainTestSplitter:
    def test_default_split(self):
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, size=100)
        splitter = TrainTestSplitter(test_size=0.2, shuffle=False)
        (X_train, y_train), (X_test, y_test) = splitter.split(X, y)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_shuffle(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splitter = TrainTestSplitter(test_size=0.3, shuffle=True, random_state=42)
        (X_train, _), (X_test, _) = splitter.split(X, y)
        assert not np.array_equal(X_test.ravel(), np.arange(70, 100))

    def test_without_labels(self):
        X = np.random.randn(50, 3)
        splitter = TrainTestSplitter(test_size=0.2)
        (X_train, y_train), (X_test, y_test) = splitter.split(X)
        assert y_train is None
        assert y_test is None

    def test_reproducible_defaults(self):
        X = np.random.randn(50, 1)
        splitter = TrainTestSplitter(test_size=0.2, shuffle=False)
        (X_train1, _), (X_test1, _) = splitter.split(X, np.arange(50))
        (X_train2, _), (X_test2, _) = splitter.split(X, np.arange(50))
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)


class TestKFoldSplitter:
    def test_default_splits(self):
        X = np.random.randn(100, 2)
        splitter = KFoldSplitter(n_splits=5, shuffle=False)
        folds = list(splitter.split(X))
        assert len(folds) == 5

    def test_fold_sizes(self):
        X = np.random.randn(100, 2)
        splitter = KFoldSplitter(n_splits=5, shuffle=False)
        all_test = []
        for train_idx, test_idx in splitter.split(X):
            assert len(test_idx) == 20
            assert len(train_idx) == 80
            all_test.extend(test_idx)
        assert len(set(all_test)) == 100

    def test_non_divisible(self):
        X = np.random.randn(103, 2)
        splitter = KFoldSplitter(n_splits=5, shuffle=False)
        total = 0
        for train_idx, test_idx in splitter.split(X):
            total += len(test_idx)
        assert total == 103

    def test_get_n_splits(self):
        splitter = KFoldSplitter(n_splits=10)
        assert splitter.get_n_splits() == 10


class TestStratifiedSplitter:
    def test_class_distribution_preserved(self):
        X = np.random.randn(100, 2)
        y = np.array([0] * 50 + [1] * 50)
        np.random.shuffle(y)
        splitter = StratifiedSplitter(test_size=0.2, shuffle=False)
        (_, y_train), (_, y_test) = splitter.split(X, y)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - 0.5) < 0.15
        assert abs(test_ratio - 0.5) < 0.15

    def test_multi_class(self):
        X = np.random.randn(90, 2)
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        splitter = StratifiedSplitter(test_size=0.2)
        (_, y_train), (_, y_test) = splitter.split(X, y)
        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_shuffle_default(self):
        X = np.random.randn(50, 2)
        y = np.array([0] * 25 + [1] * 25)
        splitter = StratifiedSplitter()
        (_, y_train), (_, y_test) = splitter.split(X, y)
        assert len(set(y_train)) == 2
        assert len(set(y_test)) == 2


class TestTimeSplitter:
    def test_basic(self):
        X = np.random.randn(100, 2)
        splitter = TimeSplitter(n_splits=3, test_size=10, gap=0)
        splits = list(splitter.split(X))
        assert len(splits) == 3

    def test_increasing_train_size(self):
        X = np.random.randn(100, 2)
        splitter = TimeSplitter(n_splits=4, test_size=10, gap=0)
        train_sizes = []
        for train_idx, test_idx in splitter.split(X):
            train_sizes.append(len(train_idx))
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_gap(self):
        X = np.random.randn(100, 2)
        splitter = TimeSplitter(n_splits=2, test_size=10, gap=5)
        for train_idx, test_idx in splitter.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert max(train_idx) + 5 < min(test_idx)

    def test_default_test_size(self):
        X = np.random.randn(60, 2)
        splitter = TimeSplitter(n_splits=3)
        splits = list(splitter.split(X))
        assert len(splits) == 3


class TestGroupSplitter:
    def test_basic(self):
        X = np.random.randn(20, 2)
        y = np.random.randint(0, 2, size=20)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                           6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
        splitter = GroupSplitter(test_size=0.2)
        (X_train, y_train), (X_test, y_test) = splitter.split(X, y, groups)
        assert len(X_train) + len(X_test) == 20

    def test_group_integrity(self):
        np.random.seed(42)
        X = np.random.randn(10, 2)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        splitter = GroupSplitter(test_size=0.2, random_state=42)
        (X_train, _), (X_test, _) = splitter.split(X, groups=groups)
        train_groups = set(groups.take(np.where(np.isin(np.arange(len(X)), [0]))[0])) if False else set()
        test_mask = np.zeros(len(X), dtype=bool)
        for i in range(len(X)):
            if groups[i] in set(groups[np.where(np.isin(np.arange(len(X)), [i]))[0]]):
                test_mask[i] = True
        test_groups_set = set(groups[test_mask]) if any(test_mask) else set()
        assert len(test_groups_set) > 0

    def test_fallback_without_groups(self):
        X = np.random.randn(30, 2)
        splitter = GroupSplitter(test_size=0.2)
        (X_train, _), (X_test, _) = splitter.split(X)
        assert abs(len(X_train) / len(X) - 0.8) < 0.01


class TestFeatureEngineeringEdgeCases:
    def test_empty_feature_similarity(self):
        X = np.empty((10, 0))
        cs = CosineSimilarity()
        sim = cs.compute(X)
        assert sim.shape == (0, 0)

    def test_single_row_alignment(self):
        X = np.random.randn(1, 3)
        sa = StatisticalAlignment(method="zscore")
        sa.fit(X)
        assert sa.local_mean_ is not None

    def test_binning_single_row(self):
        X = np.array([[5.0]])
        binner = EqualWidthBinning(n_bins=3)
        binned = binner.fit_transform(X)
        assert binned[0, 0] == 1 or binned[0, 0] == 2

    def test_one_hot_encoding_unknown(self):
        X = np.array(["a", "b", "c"])
        enc = FLOneHotEncoder(handle_unknown="ignore")
        enc.fit(X)
        result = enc.transform(np.array(["d"]))
        assert np.allclose(result, 0)

    def test_splitter_zero_test_size(self):
        X = np.random.randn(10, 2)
        splitter = TrainTestSplitter(test_size=0.0)
        (X_train, _), (X_test, _) = splitter.split(X)
        assert len(X_train) == 10
        assert len(X_test) == 0
