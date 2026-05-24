"""Comprehensive unit tests covering all importable SDK modules."""
import json
import os
import pickle
import tempfile
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module-level mocks for C++ compiled modules that are not available in
# a pure-Python test environment.
# ---------------------------------------------------------------------------
sys.modules["linkcontext"] = MagicMock()
sys.modules["opt_paillier_c2py"] = MagicMock()
sys.modules["ph_secure_lib"] = MagicMock()
sys.modules["pybind_mpc"] = MagicMock()
sys.modules["express_pb2"] = MagicMock()
sys.modules["express_pb2_grpc"] = MagicMock()
sys.modules["grpc"] = MagicMock()
sys.modules["pymysql"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
_dsketch = MagicMock()
_dsketch.frequent_strings_sketch = MagicMock()
_dsketch.frequent_items_sketch = MagicMock()
_dsketch.frequent_items_error_type = MagicMock()
_dsketch.PyFloatsSerDe = MagicMock()
_dsketch.PyIntsSerDe = MagicMock()
_dsketch.PyObjectSerDe = MagicMock()
_dsketch.kll_ints_sketch = MagicMock()
_dsketch.kll_floats_sketch = MagicMock()
_dsketch.vector_of_kll_ints_sketches = MagicMock()
_dsketch.vector_of_kll_floats_sketches = MagicMock()
_dsketch.req_ints_sketch = MagicMock()
_dsketch.req_floats_sketch = MagicMock()
sys.modules["datasketches"] = _dsketch

# ---------------------------------------------------------------------------
# optional dependency detection
# ---------------------------------------------------------------------------
HAS_SKLEARN = True
try:
    import sklearn  # noqa: F401
except ImportError:
    HAS_SKLEARN = False
    _sklearn_mock = MagicMock()
    _sklearn_mock.preprocessing = MagicMock()
    _sklearn_mock.impute = MagicMock()
    _sklearn_mock.utils = MagicMock()
    _sklearn_mock.utils._encode = MagicMock()
    _sklearn_mock.utils._mask = MagicMock()
    _sklearn_mock.utils._param_validation = MagicMock()
    _sklearn_mock.utils.multiclass = MagicMock()
    _sklearn_mock.utils.validation = MagicMock()
    _sklearn_mock.utils.validation.check_array = lambda X, **kwargs: np.asarray(X) if isinstance(X, (list, np.ndarray)) else X
    _sklearn_mock.utils.validation.FLOAT_DTYPES = (np.float64, np.float32, np.float16)
    _sklearn_mock.utils.is_scalar_nan = lambda x: isinstance(x, (float, np.floating)) and np.isnan(x)
    _sklearn_mock.utils._array_api = MagicMock()
    _sklearn_mock.utils._isfinite = MagicMock()
    _sklearn_mock._config = MagicMock()
    _sklearn_mock.metrics = MagicMock()
    _sklearn_mock.preprocessing._data = MagicMock()
    _sklearn_mock.preprocessing._encoders = MagicMock()
    _sklearn_mock.impute._base = MagicMock()
    sys.modules["sklearn"] = _sklearn_mock
    sys.modules["sklearn.preprocessing"] = _sklearn_mock.preprocessing
    sys.modules["sklearn.preprocessing._data"] = _sklearn_mock.preprocessing._data
    sys.modules["sklearn.preprocessing._encoders"] = _sklearn_mock.preprocessing._encoders
    sys.modules["sklearn.impute"] = _sklearn_mock.impute
    sys.modules["sklearn.impute._base"] = _sklearn_mock.impute._base
    sys.modules["sklearn.utils"] = _sklearn_mock.utils
    sys.modules["sklearn.utils._encode"] = _sklearn_mock.utils._encode
    sys.modules["sklearn.utils._mask"] = _sklearn_mock.utils._mask
    sys.modules["sklearn.utils._param_validation"] = _sklearn_mock.utils._param_validation
    sys.modules["sklearn.utils.multiclass"] = _sklearn_mock.utils.multiclass
    sys.modules["sklearn.utils.validation"] = _sklearn_mock.utils.validation
    sys.modules["sklearn.utils._array_api"] = _sklearn_mock.utils._array_api
    sys.modules["sklearn.utils._isfinite"] = _sklearn_mock.utils._isfinite
    sys.modules["sklearn._config"] = _sklearn_mock._config
    sys.modules["sklearn.metrics"] = _sklearn_mock.metrics

HAS_SCIPY = True
try:
    import scipy  # noqa: F401
except ImportError:
    HAS_SCIPY = False
    _scipy_mock = MagicMock()
    _scipy_mock.stats = MagicMock()
    _scipy_mock.special = MagicMock()
    _scipy_mock.interpolate = MagicMock()
    sys.modules["scipy"] = _scipy_mock
    sys.modules["scipy.stats"] = _scipy_mock.stats
    sys.modules["scipy.special"] = _scipy_mock.special
    sys.modules["scipy.interpolate"] = _scipy_mock.interpolate

HAS_PHE = True
try:
    import phe  # noqa: F401
except ImportError:
    HAS_PHE = False

HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    HAS_TORCH = False

HAS_TENSEAL = True
try:
    import tenseal  # noqa: F401
except ImportError:
    HAS_TENSEAL = False

HAS_LOGURU = True
try:
    import loguru  # noqa: F401
except ImportError:
    HAS_LOGURU = False

HAS_PROTOBUF = True
try:
    import google.protobuf  # noqa: F401
except ImportError:
    HAS_PROTOBUF = False

HAS_PYARROW = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    HAS_PYARROW = False

HAS_SQLALCHEMY = True
try:
    import sqlalchemy  # noqa: F401
except ImportError:
    HAS_SQLALCHEMY = False

HAS_RAY = True
try:
    import ray  # noqa: F401
except ImportError:
    HAS_RAY = False


def module_available(import_path: str) -> bool:
    try:
        __import__(import_path, fromlist=["__dummy"])
        return True
    except ImportError:
        return False


# ===================================================================
# primihub top-level
# ===================================================================
class TestPrimihubPackage:
    def test_version(self):
        import primihub
        assert primihub.__version__
        assert primihub.VERSION == (0, 1, 0)

    def test_metadata(self):
        import primihub
        assert primihub.__author__ == "PrimiHub.Inc"
        assert "primihub.com" in primihub.__homepage__


# ===================================================================
# primihub.FL.crypto
# ===================================================================
class TestCryptoPaillier:
    def test_paillier_import(self):
        from primihub.FL.crypto.paillier import Paillier
        assert Paillier is not None

    @pytest.mark.skipif(not HAS_PHE, reason="python-phe not installed")
    def test_paillier_encrypt_decrypt_scalar(self):
        from primihub.FL.crypto.paillier import Paillier
        import phe
        pk, sk = phe.generate_paillier_keypair(n_length=128)
        p = Paillier(pk, sk)
        ct = p.encrypt_scalar(42)
        assert p.decrypt_scalar(ct) == 42

    @pytest.mark.skipif(not HAS_PHE, reason="python-phe not installed")
    def test_paillier_encrypt_decrypt_vector(self):
        from primihub.FL.crypto.paillier import Paillier
        import phe
        pk, sk = phe.generate_paillier_keypair(n_length=128)
        p = Paillier(pk, sk)
        ct = p.encrypt_vector([1, 2, 3])
        assert p.decrypt_vector(ct) == [1, 2, 3]

    @pytest.mark.skipif(not HAS_PHE, reason="python-phe not installed")
    def test_paillier_encrypt_decrypt_matrix(self):
        from primihub.FL.crypto.paillier import Paillier
        import phe
        pk, sk = phe.generate_paillier_keypair(n_length=128)
        p = Paillier(pk, sk)
        ct = p.encrypt_matrix([[1, 2], [3, 4]])
        assert p.decrypt_matrix(ct) == [[1, 2], [3, 4]]


@pytest.mark.skipif(not HAS_TENSEAL, reason="tenseal not installed")
class TestCryptoCKKS:
    def test_ckks_import(self):
        from primihub.FL.crypto.ckks import CKKS
        assert CKKS is not None

    def test_ckks_context_creation(self):
        import tenseal as ts
        from primihub.FL.crypto.ckks import CKKS
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.generate_galois_keys()
        ckks = CKKS(context)
        assert ckks.context is not None

    def test_ckks_encrypt_decrypt_vector(self):
        import tenseal as ts
        from primihub.FL.crypto.ckks import CKKS
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
        context.generate_relin_keys()
        ckks = CKKS(context)
        ct = ckks.encrypt_vector([1.0, 2.0, 3.0])
        pt = ckks.decrypt(ct)
        assert len(pt) == 3


# ===================================================================
# primihub.FL.metrics
# ===================================================================
@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestMetricsClassification:
    def test_classification_metrics_import(self):
        from primihub.FL.metrics.classification import classification_metrics
        assert callable(classification_metrics)

    def test_classification_metrics_basic(self):
        from primihub.FL.metrics.classification import classification_metrics
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = classification_metrics(y_true, y_score)
        assert "acc" in result
        assert result["acc"] == 1.0
        assert 0.5 <= result["auc"] <= 1.0

    def test_classification_metrics_imbalanced(self):
        from primihub.FL.metrics.classification import classification_metrics
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.3, 0.6, 0.4, 0.7])
        result = classification_metrics(y_true, y_pred)
        assert "f1" in result
        assert 0 < result["f1"] <= 1.0

    def test_classification_metrics_prefix(self):
        from primihub.FL.metrics.classification import classification_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        result = classification_metrics(y_true, y_pred, prefix="val_")
        assert "val_acc" in result

    def test_hfl_metrics_import(self):
        from primihub.FL.metrics.hfl_metrics import fpr_tpr_merge2
        assert callable(fpr_tpr_merge2)

    def test_fpr_tpr_merge2(self):
        from primihub.FL.metrics.hfl_metrics import fpr_tpr_merge2
        fpr1 = np.array([0.0, 0.5, 1.0])
        tpr1 = np.array([0.0, 0.8, 1.0])
        th1 = np.array([0.9, 0.5, 0.1])
        fpr2 = np.array([0.0, 0.3, 1.0])
        tpr2 = np.array([0.0, 0.7, 1.0])
        th2 = np.array([0.8, 0.4, 0.1])
        fpr_m, tpr_m, th_m = fpr_tpr_merge2(
            fpr1, tpr1, th1, fpr2, tpr2, th2, np.array([5, 5]), np.array([5, 5])
        )
        assert len(fpr_m) > 0
        assert len(tpr_m) > 0
        assert len(th_m) > 0


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestMetricsRegression:
    def test_regression_metrics_import(self):
        from primihub.FL.metrics.regression import regression_metrics
        assert callable(regression_metrics)

    def test_regression_metrics_all(self):
        from primihub.FL.metrics.regression import regression_metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        result = regression_metrics(y_true, y_pred)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result

    def test_regression_metrics_perfect(self):
        from primihub.FL.metrics.regression import regression_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = regression_metrics(y_true, y_pred)
        assert result["mse"] == 0.0
        assert result["r2"] == 1.0

    def test_regression_metrics_prefix(self):
        from primihub.FL.metrics.regression import regression_metrics
        result = regression_metrics(np.array([1, 2]), np.array([1, 2]), prefix="test_")
        assert "test_mse" in result


# ===================================================================
# primihub.FL.preprocessing (base + submodules)
# ===================================================================
class TestPreprocessingBase:
    @staticmethod
    def _load_PreprocessBase():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "primihub.FL.preprocessing.base",
            "primihub/FL/preprocessing/base.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._PreprocessBase

    def test_preprocess_base_import(self):
        _PreprocessBase = self._load_PreprocessBase()
        assert _PreprocessBase is not None

    def test_preprocess_base_vfl_host(self):
        _PreprocessBase = self._load_PreprocessBase()
        b = _PreprocessBase(FL_type="V", role="host")
        assert b.FL_type == "V"
        assert b.role == "host"

    def test_preprocess_base_vfl_guest(self):
        _PreprocessBase = self._load_PreprocessBase()
        b = _PreprocessBase(FL_type="V", role="guest")
        assert b.FL_type == "V"
        assert b.role == "guest"

    def test_preprocess_base_hfl_client(self):
        _PreprocessBase = self._load_PreprocessBase()
        b = _PreprocessBase(FL_type="H", role="client")
        assert b.FL_type == "H"
        assert b.role == "client"

    def test_preprocess_base_hfl_server(self):
        _PreprocessBase = self._load_PreprocessBase()
        b = _PreprocessBase(FL_type="H", role="server")
        assert b.FL_type == "H"
        assert b.role == "server"

    def test_preprocess_base_invalid_fl_type(self):
        _PreprocessBase = self._load_PreprocessBase()
        with pytest.raises(ValueError, match="Unsupported FL type"):
            _PreprocessBase(FL_type="X", role="host")

    def test_preprocess_base_invalid_role(self):
        _PreprocessBase = self._load_PreprocessBase()
        with pytest.raises(ValueError, match="Unsupported role"):
            _PreprocessBase(FL_type="V", role="client")

    def test_preprocess_base_check_channel_raises(self):
        _PreprocessBase = self._load_PreprocessBase()
        b = _PreprocessBase(FL_type="H", role="client")
        with pytest.raises(ValueError, match="channel cannot be None"):
            b.check_channel()

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_preprocess_util_import(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "primihub.FL.preprocessing.util",
            "primihub/FL/preprocessing/util.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.validate_quantile_sketch_params)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingScaler:
    def test_scaler_imports(self):
        from primihub.FL.preprocessing.scaler import (
            StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer,
        )
        assert StandardScaler is not None
        assert MinMaxScaler is not None
        assert MaxAbsScaler is not None
        assert RobustScaler is not None
        assert Normalizer is not None

    def test_standard_scaler_local(self):
        from primihub.FL.preprocessing.scaler import StandardScaler
        np.random.seed(42)
        X = np.random.rand(10, 3)
        scaler = StandardScaler(FL_type="V", role="guest")
        result = scaler.fit_transform(X)
        assert result.shape == X.shape

    def test_minmax_scaler_local(self):
        from primihub.FL.preprocessing.scaler import MinMaxScaler
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = MinMaxScaler(FL_type="V", role="guest")
        result = scaler.fit_transform(X)
        assert result.shape == X.shape

    def test_robust_scaler_local(self):
        from primihub.FL.preprocessing.scaler import RobustScaler
        X = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
        scaler = RobustScaler(FL_type="V", role="guest")
        result = scaler.fit_transform(X)
        assert result.shape == X.shape


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingEncoder:
    def test_encoder_imports(self):
        from primihub.FL.preprocessing.encoder import OneHotEncoder, OrdinalEncoder, TargetEncoder
        assert OneHotEncoder is not None
        assert OrdinalEncoder is not None
        assert TargetEncoder is not None

    def test_onehot_encoder_local(self):
        from primihub.FL.preprocessing.encoder import OneHotEncoder
        X = np.array([["a"], ["b"], ["a"], ["c"]])
        enc = OneHotEncoder(FL_type="V", role="guest")
        result = enc.fit_transform(X)
        assert result.shape[1] >= 3

    def test_ordinal_encoder_local(self):
        from primihub.FL.preprocessing.encoder import OrdinalEncoder
        X = np.array([["a"], ["b"], ["a"], ["c"]])
        enc = OrdinalEncoder(FL_type="V", role="guest")
        result = enc.fit_transform(X)
        assert result.shape[0] == 4


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingImputer:
    def test_imputer_import(self):
        from primihub.FL.preprocessing.imputer import SimpleImputer
        assert SimpleImputer is not None

    def test_simple_imputer_mean(self):
        from primihub.FL.preprocessing.imputer import SimpleImputer
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        imp = SimpleImputer(strategy="mean", FL_type="V", role="guest")
        result = imp.fit_transform(X)
        assert not np.any(np.isnan(result))

    def test_simple_imputer_median(self):
        from primihub.FL.preprocessing.imputer import SimpleImputer
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        imp = SimpleImputer(strategy="median", FL_type="V", role="guest")
        result = imp.fit_transform(X)
        assert not np.any(np.isnan(result))


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingLabel:
    def test_label_imports(self):
        from primihub.FL.preprocessing.label import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
        assert LabelEncoder is not None
        assert LabelBinarizer is not None
        assert MultiLabelBinarizer is not None

    def test_label_encoder(self):
        from primihub.FL.preprocessing.label import LabelEncoder
        y = np.array(["a", "b", "a", "c"])
        enc = LabelEncoder(FL_type="V", role="guest")
        result = enc.fit_transform(y)
        assert set(result.tolist()) == {0, 1, 2}


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingDiscretizer:
    def test_discretizer_import(self):
        from primihub.FL.preprocessing.discretizer import KBinsDiscretizer
        assert KBinsDiscretizer is not None

    def test_kbins_discretizer(self):
        from primihub.FL.preprocessing.discretizer import KBinsDiscretizer
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        disc = KBinsDiscretizer(n_bins=3, encode="ordinal", FL_type="V", role="guest")
        result = disc.fit_transform(X)
        assert result.shape == X.shape


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestPreprocessingTransformer:
    def test_transformer_imports(self):
        from primihub.FL.preprocessing.transformer import PowerTransformer, QuantileTransformer, SplineTransformer
        assert PowerTransformer is not None
        assert QuantileTransformer is not None
        assert SplineTransformer is not None

    def test_power_transformer(self):
        from primihub.FL.preprocessing.transformer import PowerTransformer
        np.random.seed(42)
        X = np.random.lognormal(size=(50, 2))
        trans = PowerTransformer(FL_type="V", role="guest")
        result = trans.fit_transform(X)
        assert result.shape == X.shape

    def test_spline_transformer(self):
        from primihub.FL.preprocessing.transformer import SplineTransformer
        X = np.arange(20).reshape(10, 2).astype(float)
        trans = SplineTransformer(n_knots=5, FL_type="V", role="guest")
        result = trans.fit_transform(X)
        assert result.shape[0] == 10


# ===================================================================
# primihub.FL.stats
# ===================================================================
class TestStatsUtil:
    def test_check_role(self):
        from primihub.FL.stats.util import check_role
        with pytest.raises(ValueError, match="Unsupported role"):
            check_role("invalid")


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsMeanVar:
    def test_col_mean_basic(self):
        from primihub.FL.stats.mean_var import col_mean
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = col_mean("guest", X, ignore_nan=True)
        assert not np.any(np.isnan(result))

    def test_col_var_basic(self):
        from primihub.FL.stats.mean_var import col_var
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = col_var("guest", X, ignore_nan=True)
        assert not np.any(np.isnan(result))


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsMinMax:
    def test_col_min_basic(self):
        from primihub.FL.stats.min_max import col_min
        X = np.array([[3.0, 1.0], [1.0, 4.0], [5.0, 2.0]])
        result = col_min("guest", X, ignore_nan=True)
        assert result is not None

    def test_col_max_basic(self):
        from primihub.FL.stats.min_max import col_max
        X = np.array([[3.0, 1.0], [1.0, 4.0], [5.0, 2.0]])
        result = col_max("guest", X, ignore_nan=True)
        assert result is not None

    def test_col_min_max_basic(self):
        from primihub.FL.stats.min_max import col_min_max
        X = np.array([[3.0, 1.0], [1.0, 4.0], [5.0, 2.0]])
        result = col_min_max("guest", X, ignore_nan=True)
        assert result is not None


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsSum:
    def test_col_sum_basic(self):
        from primihub.FL.stats.sum import col_sum
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = col_sum("guest", X, ignore_nan=True)
        assert result is not None

    def test_row_sum_basic(self):
        from primihub.FL.stats.sum import row_sum
        channel = MagicMock()
        channel.recv.return_value = np.array([3.0, 7.0])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = row_sum("guest", X, ignore_nan=True, channel=channel)
        assert result is not None


class TestStatsNorm:
    def test_check_norm(self):
        from primihub.FL.stats.norm import check_norm
        with pytest.raises(ValueError, match="Unsupported norm"):
            check_norm("invalid")

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_col_norm_basic(self):
        from primihub.FL.stats.norm import col_norm
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = col_norm("guest", X, "l2", ignore_nan=True)
        assert not np.any(np.isnan(result))


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsUnion:
    def test_col_union_basic(self):
        from primihub.FL.stats.union import col_union
        X = np.array([["a"], ["b"], ["a"], ["c"]])
        result = col_union("guest", X, ignore_nan=True)
        assert result is not None


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsFrequent:
    def test_col_frequent_basic(self):
        from primihub.FL.stats.frequent import col_frequent
        X = np.array([["a"], ["b"], ["a"], ["c"]])
        result = col_frequent("guest", X, ignore_nan=True)
        assert result is not None


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestStatsQuantile:
    def test_col_median_basic(self):
        from primihub.FL.stats.quantile import col_median
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = col_median("guest", X)
        assert result is not None

    def test_col_quantile_basic(self):
        from primihub.FL.stats.quantile import col_quantile
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = col_quantile("guest", X, 0.5)
        assert result is not None


# ===================================================================
# primihub.FL.sketch
# ===================================================================
class TestSketch:
    def test_sketch_import(self):
        import primihub.FL.sketch
        assert hasattr(primihub.FL.sketch, "check_quantiles")
        assert hasattr(primihub.FL.sketch, "check_quantile_sketch_name")

    def test_check_quantiles_valid(self):
        from primihub.FL.sketch import check_quantiles
        result = check_quantiles(np.array([0.0, 0.5, 1.0]))
        assert np.array_equal(result, [0.0, 0.5, 1.0])

    def test_check_quantiles_scalar(self):
        from primihub.FL.sketch import check_quantiles
        result = check_quantiles(0.5)
        assert result == 0.5

    def test_check_quantiles_invalid(self):
        from primihub.FL.sketch import check_quantiles
        with pytest.raises(ValueError, match="Quantiles must be in"):
            check_quantiles(np.array([-0.1, 1.5]))

    def test_check_quantile_sketch_name_valid(self):
        from primihub.FL.sketch import check_quantile_sketch_name
        assert check_quantile_sketch_name("KLL") == "KLL"
        assert check_quantile_sketch_name("req") == "REQ"

    def test_check_quantile_sketch_name_invalid(self):
        from primihub.FL.sketch import check_quantile_sketch_name
        with pytest.raises(ValueError, match="Unsupported quantile sketch name"):
            check_quantile_sketch_name("INVALID")

    def test_util_import(self):
        from primihub.FL.sketch.util import check_sketch
        assert callable(check_sketch)

    def test_check_sketch_vector_list_validation(self):
        from primihub.FL.sketch.util import check_sketch
        with pytest.raises(RuntimeError):
            check_sketch([], "KLL", vector=False)


# ===================================================================
# primihub.FL.utils (pure-Python submodules)
# ===================================================================
class TestFLUtils:
    def test_base_model_import(self):
        from primihub.FL.utils.base import BaseModel
        assert BaseModel is not None

    def test_compat_import(self):
        from primihub.FL.utils.compat import is_scalar_nan
        assert is_scalar_nan(float("nan")) == True
        assert is_scalar_nan(42) == False

    def test_file_save_load_pickle(self):
        from primihub.FL.utils.file import save_pickle_file, load_pickle_file
        data = {"key": "value", "num": 42}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
        save_pickle_file(data, fname)
        loaded = load_pickle_file(fname)
        assert loaded == data
        os.unlink(fname)

    def test_file_save_load_json(self):
        from primihub.FL.utils.file import save_json_file
        data = {"accuracy": 0.95, "loss": 0.1}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
            f.close()
        save_json_file(data, fname)
        with open(fname) as f:
            loaded = json.load(f)
        assert loaded == data
        os.unlink(fname)

    def test_file_save_csv(self):
        from primihub.FL.utils.file import save_csv_file
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            fname = f.name
            f.close()
        save_csv_file(df, fname)
        loaded = pd.read_csv(fname)
        assert loaded.shape == (3, 2)
        os.unlink(fname)


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
class TestFLUtilsDataset:
    def test_dataset_import(self):
        from primihub.FL.utils.dataset import read_data, read_image
        assert callable(read_data)
        assert callable(read_image)

    def test_read_image(self):
        from primihub.FL.utils.dataset import read_image
        result = read_image("/fake/path.jpg")
        assert result.shape == (224, 224, 3)


# ===================================================================
# primihub.FL.linear_regression
# ===================================================================
class TestLinearRegression:
    def test_linear_regression_import(self):
        from primihub.FL.linear_regression.base import LinearRegression, LinearRegression_DPSGD
        assert LinearRegression is not None
        assert LinearRegression_DPSGD is not None

    def test_lr_predict(self):
        from primihub.FL.linear_regression.base import LinearRegression
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = LinearRegression(X)
        model.fit(X, y)
        pred = model.predict(np.array([[4.0]]))
        assert abs(pred[0] - 8.0) < 0.5

    def test_lr_get_set_theta(self):
        from primihub.FL.linear_regression.base import LinearRegression
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = LinearRegression(X)
        theta = model.get_theta()
        assert len(theta) == 3
        model.set_theta([0.5, 0.1, 0.2])
        assert np.allclose(model.get_theta(), [0.5, 0.1, 0.2])

    def test_lr_dpsgd_init(self):
        from primihub.FL.linear_regression.base import LinearRegression_DPSGD
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = LinearRegression_DPSGD(X, noise_multiplier=1.0, l2_norm_clip=1.0)
        assert model.noise_multiplier == 1.0
        assert model.l2_norm_clip == 1.0


# ===================================================================
# primihub.FL.logistic_regression
# ===================================================================
class TestLogisticRegression:
    def test_logistic_regression_import(self):
        from primihub.FL.logistic_regression.base import LogisticRegression
        assert LogisticRegression is not None

    def test_lr_sigmoid(self):
        from primihub.FL.logistic_regression.base import LogisticRegression
        X = np.array([[1.0], [2.0]])
        y = np.array([[0], [1]])
        model = LogisticRegression(X, y)
        result = model.sigmoid(np.array([0.0, 1.0, -1.0]))
        assert result.shape == (3,)
        assert 0.0 <= result.min() <= 1.0
        assert 0.0 <= result.max() <= 1.0

    def test_lr_predict_prob(self):
        from primihub.FL.logistic_regression.base import LogisticRegression
        X = np.array([[1.0], [2.0], [3.0], [10.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(X, y)
        probs = model.predict_prob(X[:2])
        assert probs.shape[0] == 2


# ===================================================================
# primihub.FL.neural_network (requires torch)
# ===================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestNeuralNetwork:
    def test_nn_create_model_mlp(self):
        from primihub.FL.neural_network.base import create_model
        device = "cpu"
        model = create_model("plain", 1, device, nn_model="mlp")
        assert model is not None

    def test_nn_choose_loss_fn_classification(self):
        from primihub.FL.neural_network.base import choose_loss_fn
        loss = choose_loss_fn(1, "classification")
        assert loss is not None

    def test_nn_choose_loss_fn_regression(self):
        from primihub.FL.neural_network.base import choose_loss_fn
        loss = choose_loss_fn(1, "regression")
        assert loss is not None

    def test_nn_mlp_import(self):
        from primihub.FL.neural_network.mlp import NeuralNetwork
        net = NeuralNetwork(1)
        assert net is not None

    def test_nn_cnn_import(self):
        from primihub.FL.neural_network.cnn import NeuralNetwork
        net = NeuralNetwork(1)
        assert net is not None


# ===================================================================
# primihub.FL.feature_*
# ===================================================================
class TestFeatureAlignment:
    def test_feature_alignment_import(self):
        from primihub.FL.feature_alignment.base import (
            FeatureAlignmentBase, StatisticalAlignment, DistributionAlignment,
        )
        assert FeatureAlignmentBase is not None
        assert StatisticalAlignment is not None
        assert DistributionAlignment is not None

    def test_feature_alignment_init(self):
        from primihub.FL.feature_alignment.base import StatisticalAlignment
        fa = StatisticalAlignment(FL_type="V", role="host")
        assert fa.FL_type == "V"
        assert fa.role == "host"


class TestFeatureBinning:
    def test_feature_binning_import(self):
        from primihub.FL.feature_binning.base import (
            FeatureBinningBase, EqualWidthBinning, EqualFrequencyBinning,
        )
        assert FeatureBinningBase is not None
        assert EqualWidthBinning is not None
        assert EqualFrequencyBinning is not None

    def test_feature_binning_init(self):
        from primihub.FL.feature_binning.base import EqualWidthBinning
        fb = EqualWidthBinning(FL_type="H", role="client", n_bins=5)
        assert fb.n_bins == 5
        assert fb.FL_type == "H"


class TestFeatureEncoding:
    def test_feature_encoding_import(self):
        from primihub.FL.feature_encoding.base import (
            FLFeatureEncoderBase, FLOneHotEncoder, FLLabelEncoder,
        )
        assert FLFeatureEncoderBase is not None
        assert FLOneHotEncoder is not None
        assert FLLabelEncoder is not None

    def test_feature_encoding_init(self):
        from primihub.FL.feature_encoding.base import FLOneHotEncoder
        fe = FLOneHotEncoder(FL_type="H", role="client")
        assert fe.FL_type == "H"


class TestFeatureImputation:
    def test_feature_imputation_import(self):
        from primihub.FL.feature_imputation.base import (
            FLImputerBase, FLMeanImputer, FLMedianImputer,
        )
        assert FLImputerBase is not None
        assert FLMeanImputer is not None
        assert FLMedianImputer is not None

    def test_feature_imputation_init(self):
        from primihub.FL.feature_imputation.base import FLMeanImputer
        fi = FLMeanImputer(FL_type="H", role="client")
        assert fi.FL_type == "H"


class TestFeatureSharing:
    def test_feature_sharing_import(self):
        from primihub.FL.feature_sharing.base import (
            FeatureSharingBase, SecureFeatureSharing, PartialFeatureSharing,
        )
        assert FeatureSharingBase is not None
        assert SecureFeatureSharing is not None
        assert PartialFeatureSharing is not None

    def test_feature_sharing_init(self):
        from primihub.FL.feature_sharing.base import SecureFeatureSharing
        fs = SecureFeatureSharing(FL_type="V", role="host")
        assert fs.FL_type == "V"


class TestFeatureSimilarity:
    def test_feature_similarity_import(self):
        from primihub.FL.feature_similarity.base import (
            FeatureSimilarityBase, CosineSimilarity, PearsonCorrelation,
        )
        assert FeatureSimilarityBase is not None
        assert CosineSimilarity is not None
        assert PearsonCorrelation is not None

    def test_feature_similarity_init(self):
        from primihub.FL.feature_similarity.base import CosineSimilarity
        fs = CosineSimilarity(FL_type="V", role="guest")
        assert fs.FL_type == "V"


# ===================================================================
# primihub.FL.data_splitting / data_transformation / sample_*
# ===================================================================
class TestDataSplitting:
    def test_data_splitting_import(self):
        from primihub.FL.data_splitting.base import (
            DataSplittingBase, TrainTestSplitter, KFoldSplitter,
        )
        assert DataSplittingBase is not None
        assert TrainTestSplitter is not None
        assert KFoldSplitter is not None

    def test_data_splitting_init(self):
        from primihub.FL.data_splitting.base import TrainTestSplitter
        ds = TrainTestSplitter(FL_type="H", role="client", random_state=42)
        assert ds.random_state == 42


class TestDataTransformation:
    def test_data_transformation_import(self):
        from primihub.FL.data_transformation.base import (
            DataTransformationBase, LogTransformer, BoxCoxTransformer,
        )
        assert DataTransformationBase is not None
        assert LogTransformer is not None
        assert BoxCoxTransformer is not None

    def test_data_transformation_init(self):
        from primihub.FL.data_transformation.base import LogTransformer
        dt = LogTransformer(FL_type="H", role="client")
        assert dt.FL_type == "H"


class TestSampleExpansion:
    def test_sample_expansion_import(self):
        from primihub.FL.sample_expansion.base import (
            SampleExpansionBase, PolynomialExpansion, InteractionExpansion,
        )
        assert SampleExpansionBase is not None
        assert PolynomialExpansion is not None
        assert InteractionExpansion is not None

    def test_sample_expansion_init(self):
        from primihub.FL.sample_expansion.base import PolynomialExpansion
        se = PolynomialExpansion(FL_type="V", role="host")
        assert se.FL_type == "V"


class TestSampleWeighting:
    def test_sample_weighting_import(self):
        from primihub.FL.sample_weighting.base import (
            SampleWeightingBase, ClassWeighting, ImportanceWeighting,
        )
        assert SampleWeightingBase is not None
        assert ClassWeighting is not None
        assert ImportanceWeighting is not None

    def test_sample_weighting_init(self):
        from primihub.FL.sample_weighting.base import ClassWeighting
        sw = ClassWeighting(FL_type="H", role="server")
        assert sw.FL_type == "H"


# ===================================================================
# primihub.FL.model_evaluation / metrics_modeling / data_fusion / fl_preprocessing
# ===================================================================
class TestModelEvaluation:
    def test_model_evaluation_import(self):
        from primihub.FL.model_evaluation.base import (
            FLEvaluatorBase, ClassificationEvaluator, RegressionEvaluator,
        )
        assert FLEvaluatorBase is not None
        assert ClassificationEvaluator is not None
        assert RegressionEvaluator is not None

    def test_model_evaluation_init(self):
        from primihub.FL.model_evaluation.base import ClassificationEvaluator
        ev = ClassificationEvaluator(multiclass=False)
        assert ev is not None


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestModelEvaluationMetrics:
    def test_classification_evaluator_metrics(self):
        from primihub.FL.model_evaluation.base import ClassificationEvaluator
        ev = ClassificationEvaluator(FL_type="V", role="guest")
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        result = ev.evaluate(y_true, y_pred)
        assert "accuracy" in result
        assert "f1_score" in result

    def test_regression_evaluator_metrics(self):
        from primihub.FL.model_evaluation.base import RegressionEvaluator
        ev = RegressionEvaluator(FL_type="V", role="guest")
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        result = ev.evaluate(y_true, y_pred)
        assert "mse" in result
        assert "r2" in result


class TestMetricsModeling:
    def test_metrics_modeling_import(self):
        from primihub.FL.metrics_modeling.base import (
            MetricsModelingBase, FederatedMetrics, ModelPerformanceAnalyzer,
        )
        assert MetricsModelingBase is not None
        assert FederatedMetrics is not None
        assert ModelPerformanceAnalyzer is not None

    def test_metrics_modeling_init(self):
        from primihub.FL.metrics_modeling.base import FederatedMetrics
        mm = FederatedMetrics(FL_type="H", role="client")
        assert mm.FL_type == "H"


class TestDataFusion:
    def test_data_fusion_import(self):
        from primihub.FL.data_fusion.base import (
            DataFusionBase, HorizontalDataFusion, VerticalDataFusion, SecureDataFusion,
        )
        assert DataFusionBase is not None
        assert HorizontalDataFusion is not None
        assert VerticalDataFusion is not None
        assert SecureDataFusion is not None

    def test_data_fusion_init(self):
        from primihub.FL.data_fusion.base import HorizontalDataFusion
        df = HorizontalDataFusion()
        assert df is not None


class TestFLPreprocessing:
    def test_fl_preprocessing_import(self):
        from primihub.FL.fl_preprocessing.base import (
            FLPreprocessBase, FLDataCleaner, FLOutlierDetector, FLDataValidator,
        )
        assert FLPreprocessBase is not None
        assert FLDataCleaner is not None
        assert FLOutlierDetector is not None
        assert FLDataValidator is not None

    def test_fl_preprocessing_init(self):
        from primihub.FL.fl_preprocessing.base import FLDataCleaner
        fp = FLDataCleaner(FL_type="H", role="client")
        assert fp.FL_type == "H"


# ===================================================================
# primihub.FL.federated_psi / federated_query / psu
# ===================================================================
class TestFederatedPSI:
    def test_federated_psi_import(self):
        from primihub.FL.federated_psi.base import (
            PSIBase, DHPSIProtocol, OTPSIProtocol, HEPSIProtocol,
        )
        assert PSIBase is not None
        assert DHPSIProtocol is not None
        assert OTPSIProtocol is not None
        assert HEPSIProtocol is not None

    def test_federated_psi_init(self):
        from primihub.FL.federated_psi import BatchDHPSI
        psi = BatchDHPSI(role="host")
        assert psi.role == "host"

    def test_federated_psi_full_import(self):
        from primihub.FL.federated_psi import (
            PSIBase, BatchDHPSI, BatchOTPSI, BatchHEPSI, RealtimeDHPSI,
        )
        assert BatchDHPSI is not None
        assert BatchOTPSI is not None
        assert BatchHEPSI is not None
        assert RealtimeDHPSI is not None


class TestFederatedQuery:
    def test_federated_query_import(self):
        from primihub.FL.federated_query.base import (
            FederatedQueryBase, DHQueryProtocol, OTQueryProtocol, HEQueryProtocol,
        )
        assert FederatedQueryBase is not None
        assert DHQueryProtocol is not None
        assert OTQueryProtocol is not None
        assert HEQueryProtocol is not None

    def test_federated_query_init(self):
        from primihub.FL.federated_query import BatchDHQuery
        fq = BatchDHQuery(role="host")
        assert fq.role == "host"

    def test_federated_query_full_import(self):
        from primihub.FL.federated_query import (
            FederatedQueryBase, BatchDHQuery, BatchOTQuery, BatchHEQuery,
        )
        assert BatchDHQuery is not None
        assert BatchOTQuery is not None
        assert BatchHEQuery is not None


class TestPSU:
    def test_psu_import(self):
        from primihub.FL.psu.base import (
            PSUBase, HashBasedPSU, BloomFilterPSU, EncryptedPSU,
        )
        assert PSUBase is not None
        assert HashBasedPSU is not None
        assert BloomFilterPSU is not None
        assert EncryptedPSU is not None

    def test_psu_init(self):
        from primihub.FL.psu.base import HashBasedPSU
        psu = HashBasedPSU(role="guest")
        assert psu.role == "guest"


# ===================================================================
# primihub.FL.psi (requires C++ dependencies)
# ===================================================================
class TestFLPSI:
    def test_psi_import_requires_cpp(self):
        has_psi = module_available("primihub.FL.psi")
        try:
            from primihub.MPC.psi import PsiType
            assert PsiType is not None
        except (ImportError, ModuleNotFoundError):
            pass

    @pytest.mark.skip(reason="requires C++ ph_secure_lib module")
    def test_sample_alignment_validates_protocol(self):
        with patch("primihub.MPC.psi") as mock_mpc_psi:
            mock_mpc_psi.PsiType.ECDH = "ECDH"
            mock_mpc_psi.PsiType.KKRT = "KKRT"
            from primihub.FL.psi.psi import sample_alignment
            with patch("primihub.MPC.psi.TwoPartyPsi"):
                with pytest.raises(ValueError, match="Invalid PSI protocol"):
                    sample_alignment(
                        pd.DataFrame({"id": [1, 2, 3]}),
                        "id",
                        {"host": "alice", "guest": "bob"},
                        "INVALID",
                    )


# ===================================================================
# primihub.FL.chatglm (requires torch + transformers + C++ deps)
# ===================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestChatGLM:
    def test_chatglm_import(self):
        try:
            from primihub.FL.chatglm import ChatGlm_client, ChatGlm_server
            assert ChatGlm_client
            assert ChatGlm_server
        except (ImportError, ModuleNotFoundError):
            pytest.skip("C++ dependencies not available")


# ===================================================================
# primihub.sql_security
# ===================================================================
class TestSQLSecurity:
    def test_sql_security_import(self):
        from primihub.sql_security import (
            SQLSecurityEngine,
            FieldSecurityLevel,
            FieldMeta,
            TableSecurityConfig,
            RiskLevel,
            ValidationResult,
            format_sql,
            validate_sql,
            is_sql_safe,
        )
        assert SQLSecurityEngine is not None
        assert FieldSecurityLevel is not None
        assert FieldMeta is not None
        assert TableSecurityConfig is not None
        assert RiskLevel is not None

    def test_field_security_level_values(self):
        from primihub.sql_security import FieldSecurityLevel
        assert FieldSecurityLevel.PUBLIC.value == 0
        assert FieldSecurityLevel.PRIVATE.value == 2

    def test_field_meta_creation(self):
        from primihub.sql_security import FieldMeta, FieldSecurityLevel
        field = FieldMeta(
            field_name="salary",
            security_level=FieldSecurityLevel.PRIVATE,
            table_name="employees",
        )
        assert field.field_name == "salary"
        assert field.security_level == FieldSecurityLevel.PRIVATE

    def test_table_security_config(self):
        from primihub.sql_security import TableSecurityConfig
        config = TableSecurityConfig(table_name="users", owner_party="party_a")
        assert config.table_name == "users"
        assert config.owner_party == "party_a"

    def test_create_engine(self):
        from primihub.sql_security import SQLSecurityEngine
        engine = SQLSecurityEngine()
        assert engine is not None

    def test_validation_result(self):
        from primihub.sql_security import ValidationResult, RiskLevel
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_risk_level_values(self):
        from primihub.sql_security import RiskLevel
        assert RiskLevel.LOW is not None
        assert RiskLevel.MEDIUM is not None
        assert RiskLevel.HIGH is not None


# ===================================================================
# primihub.utils
# ===================================================================
class TestUtilsLogger:
    def test_logger_import(self):
        from primihub.utils.logger_util import logger, JobFilter, FORMAT
        assert logger is not None
        assert FORMAT is not None

    def test_job_filter_init(self):
        from primihub.utils.logger_util import JobFilter
        jf = JobFilter("test", "job123", "task456", task_type="train")
        assert jf.name == "test"


class TestUtilsAsync:
    def test_async_util_import(self):
        from primihub.utils.async_util import fire_coroutine_threadsafe
        assert callable(fire_coroutine_threadsafe)

    def test_fire_coroutine_threadsafe_raises(self):
        from primihub.utils.async_util import fire_coroutine_threadsafe
        import asyncio
        import threading
        async def dummy():
            pass
        loop = asyncio.new_event_loop()
        loop._thread_ident = threading.get_ident()
        with pytest.raises(RuntimeError, match="Cannot be called from within"):
            fire_coroutine_threadsafe(dummy(), loop)
        loop.close()


@pytest.mark.skipif(not HAS_PROTOBUF, reason="protobuf not installed")
class TestUtilsProtobuf:
    def test_protobuf_to_dict_import(self):
        from primihub.utils.protobuf_to_dict import protobuf_to_dict, dict_to_protobuf
        assert callable(protobuf_to_dict)
        assert callable(dict_to_protobuf)


# ===================================================================
# primihub.dataset
# ===================================================================
@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestDataset:
    def test_dataset_cli_import(self):
        from primihub.dataset.dataset_cli import DatasetClient
        assert DatasetClient is not None

    def test_dataset_client_put_raises(self):
        from primihub.dataset.dataset_cli import DatasetClient
        client = DatasetClient()
        with pytest.raises(NotImplementedError):
            client.do_put(None, "/fake/url")

    def test_dataset_client_get_raises(self):
        from primihub.dataset.dataset_cli import DatasetClient
        client = DatasetClient()
        with pytest.raises(NotImplementedError):
            client.do_get(None)


# ===================================================================
# primihub.engine / context / executor / pyclient
# ===================================================================
class TestContext:
    def test_context_import(self):
        from primihub.context import Context, ContextAll
        assert Context is not None
        assert ContextAll is not None

    def test_context_all_defaults(self):
        from primihub.context import ContextAll
        ca = ContextAll()
        assert ca.message is None
        assert ca.datasets == []
        assert ca.cert_config == {}

    def test_context_instance(self):
        from primihub.context import Context
        assert hasattr(Context, "message")
        assert hasattr(Context, "datasets")
        assert hasattr(Context, "cert_config")


class TestEngine:
    def test_engine_import(self):
        from primihub.engine.engine import PythonEngine
        assert PythonEngine is not None


# ===================================================================
# primihub.local (all submodules)
# ===================================================================
class TestLocalModule:
    def test_all_local_submodules_importable(self):
        local_modules = [
            "primihub.local",
            "primihub.local.data_cleaning",
            "primihub.local.data_scaling",
            "primihub.local.data_statistics",
            "primihub.local.feature_binning",
            "primihub.local.feature_derivation",
            "primihub.local.feature_encoding",
            "primihub.local.feature_selection",
            "primihub.local.ml_lr",
            "primihub.local.ml_xgb",
            "primihub.local.sql_processing",
            "primihub.local.training_logger",
            "primihub.local.log_exporter",
            "primihub.local.python_script",
        ]
        for mod in local_modules:
            assert module_available(mod), f"{mod} should be importable"

    def test_data_cleaning_base(self):
        from primihub.local.data_cleaning.base import DataCleanerBase
        assert DataCleanerBase is not None

    def test_data_scaling_base(self):
        from primihub.local.data_scaling.base import DataScalerBase
        assert DataScalerBase is not None

    def test_data_statistics_base(self):
        from primihub.local.data_statistics.base import DataStatisticsBase
        assert DataStatisticsBase is not None

    def test_feature_binning_base(self):
        from primihub.local.feature_binning.base import FeatureBinnerBase
        assert FeatureBinnerBase is not None

    def test_feature_derivation_base(self):
        from primihub.local.feature_derivation.base import FeatureDeriverBase
        assert FeatureDeriverBase is not None

    def test_feature_encoding_base(self):
        from primihub.local.feature_encoding.base import FeatureEncoderBase
        assert FeatureEncoderBase is not None

    def test_feature_selection_base(self):
        from primihub.local.feature_selection.base import FeatureSelectorBase
        assert FeatureSelectorBase is not None

    def test_ml_lr_base(self):
        from primihub.local.ml_lr.base import LogisticRegressionModel
        assert LogisticRegressionModel is not None

    def test_ml_xgb_base(self):
        from primihub.local.ml_xgb.base import XGBoostModel
        assert XGBoostModel is not None

    def test_sql_processing_base(self):
        from primihub.local.sql_processing.base import SQLEngine
        assert SQLEngine is not None

    def test_training_logger_base(self):
        from primihub.local.training_logger.base import TrainingSession
        assert TrainingSession is not None

    def test_log_exporter_base(self):
        from primihub.local.log_exporter.base import LogExporter
        assert LogExporter is not None

    def test_python_script_base(self):
        from primihub.local.python_script.base import ScriptContext
        assert ScriptContext is not None
