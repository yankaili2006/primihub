# all adopted from scikit-learn
import numbers
import numpy as np
from scipy.sparse import issparse
from itertools import compress
from numbers import Integral
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    validate_parameter_constraints,
)
# 兼容性处理：不同版本的sklearn可能有不同的is_scalar_nan
try:
    from sklearn.utils import is_scalar_nan
except ImportError:
    # 如果sklearn版本没有is_scalar_nan，提供一个兼容实现
    def is_scalar_nan(x):
        """判断是否为标量NaN"""
        return isinstance(x, float) and np.isnan(x)

from sklearn.utils._encode import MissingValues, _nandict, _NaNCounter
from sklearn.utils.validation import check_array, column_or_1d
from sklearn.utils._array_api import get_namespace
from sklearn.utils._isfinite import FiniteStatus, cy_isfinite
from sklearn._config import get_config as _get_config
from contextlib import suppress


def validate_quantile_sketch_params(caller):
    parameter_constraints = {
        "sketch_name": [StrOptions({"KLL", "REQ"})],
        "k": [Interval(Integral, 1, None, closed="left")],
        "is_hra": ["boolean"],
    }
    validate_parameter_constraints(
        parameter_constraints,
        params={
            "sketch_name": caller.sketch_name,
            "k": caller.k,
            "is_hra": caller.is_hra,
        },
        caller_name=caller.__class__.__name__,
    )


def validatea_freq_sketch_params(caller):
    parameter_constraints = {
        "error_type": [StrOptions({"NFP", "NFN"})],
        "k": [Interval(Integral, 1, None, closed="left")],
    }
    validate_parameter_constraints(
        parameter_constraints,
        params={
            "sketch_name": caller.error_type,
            "k": caller.k,
        },
        caller_name=caller.__class__.__name__,
    )
