import sys
import types
import numpy as np

np.float = float

import numpy as _real_np
_real_np.float = float


class _MockDtype:
    def __init__(self, dtype):
        self._dtype = dtype
    def astype(self, _str):
        return self
    def to_dict(self):
        return {k: str(v) for k, v in self._dtype.items()} if isinstance(self._dtype, dict) else {}

class _MockDataFrame:
    def __init__(self, data=None, index=None, columns=None, dtype=None):
        if isinstance(data, dict):
            _cols = list(data.keys())
            _vals = {}
            for k in _cols:
                v = data[k]
                if isinstance(v, (list, tuple, _real_np.ndarray)):
                    try:
                        arr = _real_np.asarray([float(x) for x in v], dtype=float)
                    except (ValueError, TypeError):
                        arr = _real_np.array(v, dtype=object)
                    _vals[k] = arr
                elif isinstance(v, _MockSeries):
                    _vals[k] = v._data.copy()
                else:
                    _vals[k] = _real_np.asarray([v], dtype=float)
            self._columns = _cols
            self._data = _vals
            self._index = index
        elif data is not None:
            arr = _real_np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self._columns = [f"col_{i}" for i in range(arr.shape[1])]
            self._data = {f"col_{i}": arr[:, i] for i in range(arr.shape[1])}
            self._index = index
        else:
            self._data = {}
            self._columns = []
            self._index = index

    @property
    def columns(self):
        return self._columns
    @columns.setter
    def columns(self, val):
        self._columns = list(val)

    @property
    def dtypes(self):
        return _MockDtype({c: self._data[c].dtype for c in self._columns})
    @dtypes.setter
    def dtypes(self, val):
        pass

    @property
    def size(self):
        return _real_np.prod(self.shape)

    @property
    def shape(self):
        if not self._data:
            return (0, 0)
        return (len(next(iter(self._data.values()))), len(self._columns))
    @property
    def values(self):
        if not self._data:
            return _real_np.array([])
        return _real_np.column_stack([self._data[c] for c in self._columns])
    @property
    def empty(self):
        return len(self._columns) == 0
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        if isinstance(key, str):
            return _MockSeries(self._data[key], name=key)
        if isinstance(key, list):
            return _MockDataFrame({k: self._data[k] for k in key if k in self._columns})
        if isinstance(key, _MockSeries):
            return _MockDataFrame({c: v[key._data.astype(bool)] for c, v in self._data.items()})
        return self
    def __setitem__(self, key, val):
        if isinstance(val, _MockSeries):
            self._data[key] = val._data.copy()
        else:
            self._data[key] = _real_np.asarray(val)
        if key not in self._columns:
            self._columns.append(key)
    def __contains__(self, key):
        return key in self._columns
    def __iter__(self):
        return iter(self._columns)
    def keys(self):
        return self._columns
    def items(self):
        return [(k, v) for k, v in self._data.items()]
    def pop(self, key):
        if key in self._data:
            v = self._data.pop(key)
            self._columns = [c for c in self._columns if c != key]
            return v
        return None
    def drop(self, columns=None, **kw):
        if columns:
            new_data = {k: v for k, v in self._data.items() if k not in columns}
            new_cols = [c for c in self._columns if c not in columns]
            r = _MockDataFrame({})
            r._data = new_data
            r._columns = new_cols
            return r
        return self
    def select_dtypes(self, include=None):
        result = _MockDataFrame({})
        for k, v in self._data.items():
            if include is None:
                result._data[k] = v
                result._columns.append(k)
            elif np.issubdtype(v.dtype, np.number) if hasattr(v.dtype, 'kind') and v.dtype.kind in ('f', 'i', 'u', 'c') else False:
                result._data[k] = v
                result._columns.append(k)
        return result
    def dropna(self):
        new_data = {}
        for k, v in self._data.items():
            m = ~_real_np.isnan(v.astype(float)) if v.dtype.kind == 'f' else _real_np.ones(len(v), dtype=bool)
            new_data[k] = v[m]
        r = _MockDataFrame({})
        r._data = new_data
        r._columns = [c for c in self._columns]
        return r
    def isna(self):
        r = _MockDataFrame({})
        r._data = {k: _real_np.isnan(v.astype(float)) for k, v in self._data.items()}
        r._columns = [c for c in self._columns]
        return r
    def isnull(self):
        return self.isna()
    def sum(self, axis=None):
        if axis == 1:
            arr = _real_np.column_stack([v.astype(float) for v in self._data.values()]) if self._data else _real_np.array([]).reshape(len(next(iter(self._data.values()))), 0)
            return _MockSeries(arr.sum(axis=1))
        vals = _real_np.array([_real_np.nansum(v.astype(float)) for v in self._data.values()])
        return _MockSeries(vals)
    def mean(self, axis=None):
        return _MockSeries(_real_np.array([_real_np.nanmean(v.astype(float)) for v in self._data.values()]))
    def var(self, ddof=1):
        return _MockSeries(_real_np.array([_real_np.nanvar(v.astype(float), ddof=ddof) for v in self._data.values()]))
    def std(self, ddof=1):
        return _MockSeries(_real_np.array([_real_np.nanstd(v.astype(float), ddof=ddof) for v in self._data.values()]))
    def min(self):
        return _MockSeries(_real_np.array([_real_np.nanmin(v.astype(float)) for v in self._data.values()]))
    def max(self):
        return _MockSeries(_real_np.array([_real_np.nanmax(v.astype(float)) for v in self._data.values()]))
    def median(self):
        return _MockSeries(_real_np.array([_real_np.nanmedian(v.astype(float)) for v in self._data.values()]))
    def mode(self):
        def _mode(arr):
            a = arr[~_real_np.isnan(arr.astype(float))]
            if len(a) == 0: return _real_np.nan
            a_rounded = _real_np.round(a, 6)
            if len(a_rounded) == 0: return _real_np.nan
            vals, counts = _real_np.unique(a_rounded, return_counts=True)
            return float(vals[_real_np.argmax(counts)])
        return _MockSeries(_real_np.array([_mode(v) for v in self._data.values()]))
    def skew(self):
        from scipy.stats import skew as _skew
        def _s(v):
            a = v[~_real_np.isnan(v.astype(float))]
            if len(a) < 3: return 0.0
            return float(_skew(a))
        return _MockSeries(_real_np.array([_s(v) for v in self._data.values()]))
    def kurtosis(self):
        from scipy.stats import kurtosis as _kurt
        def _k(v):
            a = v[~_real_np.isnan(v.astype(float))]
            if len(a) < 4: return 0.0
            return float(_kurt(a))
        return _MockSeries(_real_np.array([_k(v) for v in self._data.values()]))
    def quantile(self, q):
        return _MockSeries(_real_np.array([_real_np.nanquantile(v.astype(float), q) for v in self._data.values()]))
    def nunique(self):
        return _MockSeries(_real_np.array([len(_real_np.unique(v[~_real_np.isnan(v.astype(float))])) for v in self._data.values()]))
    def corr(self, method='pearson'):
        arr = _real_np.column_stack([v.astype(float) for v in self._data.values()])
        if arr.shape[1] < 2: return _MockDataFrame({})
        cm = _real_np.corrcoef(arr.T)
        r = _MockDataFrame({})
        r._data = {self._columns[i]: _real_np.array([cm[i, j] for j in range(len(self._columns))]) for i in range(len(self._columns))}
        r._columns = [c for c in self._columns]
        r._corr_cols = self._columns[:]
        return r
    def astype(self, dtype):
        return self
    def copy(self):
        r = _MockDataFrame({})
        r._data = {k: v.copy() for k, v in self._data.items()}
        r._columns = [c for c in self._columns]
        return r
    @property
    def iloc(self):
        class _ILocIndexer:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, idx):
                if isinstance(idx, tuple):
                    row, col = idx
                    col_names = list(self._data.keys())
                    return self._data[col_names[col]][row]
                return self._data[list(self._data.keys())[idx]]
        return _ILocIndexer(self._data)
    @property
    def loc(self):
        class _LocIndexer:
            def __init__(self, data, columns):
                self._data = data
                self._columns = columns
            def __getitem__(self, idx):
                if isinstance(idx, tuple):
                    row_label, col_label = idx
                    arr = self._data[col_label]
                    if row_label in self._columns:
                        row_idx = self._columns.index(row_label)
                        if isinstance(arr, _real_np.ndarray):
                            return float(arr[row_idx])
                        return arr
                    return _MockSeries(arr)
                return self
        return _LocIndexer(self._data, self._columns)
    def to_dict(self):
        result = {}
        _corr_cols = getattr(self, '_corr_cols', None)
        for k, v in self._data.items():
            if isinstance(v, dict):
                result[k] = v
            elif _corr_cols is not None and isinstance(v, _real_np.ndarray) and len(v) == len(_corr_cols):
                result[k] = {_corr_cols[j]: float(v[j]) for j in range(len(_corr_cols))}
            else:
                result[k] = v.tolist()
        return result
    def any(self, axis=None):
        if axis == 1:
            arr = _real_np.column_stack([v.astype(float) for v in self._data.values()]) if self._data else _real_np.array([]).reshape(0, 0)
            if arr.ndim == 2 and arr.shape[1] > 0:
                return _MockSeries(arr.any(axis=1))
        return False
    def __repr__(self):
        return f"MockDataFrame(shape={self.shape}, columns={self._columns})"


class _MockSeries:
    def __init__(self, data=None, name=None, dtype=None):
        if isinstance(data, _MockSeries):
            self._data = data._data.copy()
            self.name = data.name
        elif data is not None:
            self._data = _real_np.asarray(data, dtype=dtype)
            self.name = name
        else:
            self._data = _real_np.array([])
            self.name = name

    def __len__(self):
        return len(self._data)
    def __iter__(self):
        return iter(self._data)
    def __getitem__(self, key):
        if isinstance(key, slice):
            return _MockSeries(self._data[key])
        if isinstance(key, _real_np.ndarray):
            return _MockSeries(self._data[key])
        if isinstance(key, (int, _real_np.integer)):
            return self._data[key]
        return self._data[key]
    def __gt__(self, other):
        if isinstance(other, (int, float, _real_np.floating, _real_np.integer)):
            return _MockSeries(self._data > other)
        return _MockSeries(self._data > other._data)
    def __lt__(self, other):
        if isinstance(other, (int, float, _real_np.floating, _real_np.integer)):
            return _MockSeries(self._data < other)
        return _MockSeries(self._data < other._data)
    def __ge__(self, other):
        return _MockSeries(self._data >= (other if isinstance(other, (int, float)) else other._data))
    def __le__(self, other):
        return _MockSeries(self._data <= (other if isinstance(other, (int, float)) else other._data))
    def __eq__(self, other):
        if isinstance(other, (int, float, str, _real_np.floating, _real_np.integer)):
            return _MockSeries(self._data == other)
        return _MockSeries(self._data == (other._data if isinstance(other, _MockSeries) else other))
    def __ne__(self, other):
        return _MockSeries(self._data != (other._data if isinstance(other, _MockSeries) else other))
    def __or__(self, other):
        if isinstance(other, _MockSeries):
            return _MockSeries(self._data.astype(bool) | other._data.astype(bool))
        return _MockSeries(self._data.astype(bool) | bool(other))
    def __and__(self, other):
        if isinstance(other, _MockSeries):
            return _MockSeries(self._data.astype(bool) & other._data.astype(bool))
        return _MockSeries(self._data.astype(bool) & bool(other))
    def __invert__(self):
        return _MockSeries(~self._data.astype(bool))
    def __add__(self, other):
        if isinstance(other, (int, float, _real_np.floating, _real_np.integer)):
            return _MockSeries(self._data + other)
        return _MockSeries(self._data + (other._data if isinstance(other, _MockSeries) else other))
    def __radd__(self, other):
        return _MockSeries(other + self._data)
    def __sub__(self, other):
        return _MockSeries(self._data - (other._data if isinstance(other, _MockSeries) else other))
    def __mul__(self, other):
        return _MockSeries(self._data * (other._data if isinstance(other, _MockSeries) else other))
    def __truediv__(self, other):
        return _MockSeries(self._data / (other._data if isinstance(other, _MockSeries) else other))
    def dropna(self):
        if self._data.dtype.kind in ('f', 'i', 'u', 'c'):
            m = ~_real_np.isnan(self._data.astype(float))
            return _MockSeries(self._data[m])
        m = _real_np.ones(len(self._data), dtype=bool)
        return _MockSeries(self._data[m])
    def isna(self):
        return _MockSeries(_real_np.isnan(self._data.astype(float)))
    def isnull(self):
        return self.isna()
    def sum(self):
        return _real_np.nansum(self._data.astype(float))
    def mean(self):
        return float(_real_np.nanmean(self._data.astype(float)))
    def var(self, ddof=1):
        return float(_real_np.nanvar(self._data.astype(float), ddof=ddof))
    def std(self, ddof=1):
        return float(_real_np.nanstd(self._data.astype(float), ddof=ddof))
    def min(self):
        return float(_real_np.nanmin(self._data.astype(float)))
    def max(self):
        return float(_real_np.nanmax(self._data.astype(float)))
    def median(self):
        return float(_real_np.nanmedian(self._data.astype(float)))
    def mode(self):
        vals = self._data[~_real_np.isnan(self._data.astype(float))]
        if len(vals) == 0:
            return _MockSeries(_real_np.array([]))
        unique_vals, counts = _real_np.unique(vals, return_counts=True)
        return _MockSeries(_real_np.array([float(unique_vals[_real_np.argmax(counts)])]))
    def skew(self):
        from scipy.stats import skew as _skew
        vals = self._data[~_real_np.isnan(self._data.astype(float))]
        return float(_skew(vals)) if len(vals) > 2 else 0.0
    def kurtosis(self):
        from scipy.stats import kurtosis as _kurt
        vals = self._data[~_real_np.isnan(self._data.astype(float))]
        return float(_kurt(vals)) if len(vals) > 3 else 0.0
    def quantile(self, q):
        return float(_real_np.nanquantile(self._data.astype(float), q))
    def nunique(self):
        return int(len(_real_np.unique(self._data[~_real_np.isnan(self._data.astype(float))])))
    def unique(self):
        return _real_np.unique(self._data)
    @property
    def iloc(self):
        class _ILocIndexer:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, idx):
                return self._data[idx]
        return _ILocIndexer(self._data)
        if dtype == str:
            return _MockSeries(self._data.astype(str))
        return _MockSeries(self._data.astype(dtype))
    def tolist(self):
        return self._data.tolist()
    def copy(self):
        return _MockSeries(self._data.copy(), name=self.name)
    def sum(self):
        return _real_np.nansum(self._data.astype(float))
    def any(self):
        return bool(self._data.any())
    def all(self):
        return bool(self._data.all())
    @property
    def values(self):
        return self._data
    @property
    def dtype(self):
        return self._data.dtype
    def __repr__(self):
        return f"MockSeries(shape={self._data.shape})"


class _MockPandas(types.ModuleType):
    DataFrame = _MockDataFrame
    Series = _MockSeries

_mock_pd = _MockPandas('pandas')
sys.modules['pandas'] = _mock_pd

for _sub in ('pandas.core', 'pandas.core.api', 'pandas.core.frame', 'pandas.core.series',
             'pandas.core.internals', 'pandas.core.dtypes', 'pandas.core.dtypes.common',
             'pandas.api', 'pandas.api.types', 'pandas.util', 'pandas.io',
             'pandas.io.common', 'pandas.compat', 'pandas._libs', 'pandas._libs.lib'):
    mod = _sub
    parts = mod.split('.')
    parent = sys.modules[parts[0]]
    for part in parts[1:]:
        if not hasattr(parent, part):
            setattr(parent, part, types.ModuleType(part))
        parent = getattr(parent, part)
    if mod not in sys.modules:
        sys.modules[mod] = parent
