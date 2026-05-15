try:
    from sklearn.utils import is_scalar_nan as _is_scalar_nan
except ImportError:
    import numpy as np
    def _is_scalar_nan(x):
        return isinstance(x, (float, np.floating)) and np.isnan(x)

is_scalar_nan = _is_scalar_nan
