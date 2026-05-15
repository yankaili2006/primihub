import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# These tests validate the preprocessing logic directly using sklearn.
# The FL wrappers add communication hooks on top of the same sklearn functions.
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestScaler:
    def test_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        result = scaler.fit_transform(x)
        assert result.shape == x.shape
        assert abs(result.mean()) < 1e-10
        assert abs(result.std(axis=0)[0] - 1.0) < 1e-10

    def test_minmax_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_standard_scaler_transform(self):
        from sklearn.preprocessing import StandardScaler
        x_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_test = np.array([[2.0, 3.0]])
        result = scaler.transform(x_test)
        assert result.shape == (1, 2)

    def test_robust_scaler(self):
        from sklearn.preprocessing import RobustScaler
        x = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
        scaler = RobustScaler()
        result = scaler.fit_transform(x)
        assert result.shape == x.shape


class TestImputer:
    def test_mean_imputation(self):
        from sklearn.impute import SimpleImputer
        x = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        imputer = SimpleImputer(strategy='mean')
        result = imputer.fit_transform(x)
        assert not np.any(np.isnan(result))

    def test_median_imputation(self):
        from sklearn.impute import SimpleImputer
        x = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        imputer = SimpleImputer(strategy='median')
        result = imputer.fit_transform(x)
        assert not np.any(np.isnan(result))


class TestEncoder:
    def test_onehot_encoder(self):
        from sklearn.preprocessing import OneHotEncoder
        x = np.array([['a'], ['b'], ['a'], ['c']])
        encoder = OneHotEncoder(sparse_output=False)
        result = encoder.fit_transform(x)
        assert result.shape[1] >= 3

    def test_label_encoder(self):
        from sklearn.preprocessing import LabelEncoder
        x = np.array(['a', 'b', 'a', 'c'])
        encoder = LabelEncoder()
        result = encoder.fit_transform(x)
        assert set(result.tolist()) == {0, 1, 2}


class TestDiscretizer:
    def test_kbins_discretizer(self):
        from sklearn.preprocessing import KBinsDiscretizer
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        disc = KBinsDiscretizer(n_bins=3, encode='ordinal')
        result = disc.fit_transform(x)
        assert result.shape == x.shape
        assert result.min() >= 0 and result.max() <= 2


class TestTransformer:
    def test_power_transformer(self):
        from sklearn.preprocessing import PowerTransformer
        np.random.seed(42)
        x = np.random.lognormal(size=(50, 2))
        trans = PowerTransformer(method='yeo-johnson')
        result = trans.fit_transform(x)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))
