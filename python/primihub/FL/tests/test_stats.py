import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MockChannel:
    def __init__(self):
        self.store = {}

    def send(self, key, val):
        self.store[key] = val

    def recv(self, key):
        return self.store.get(key)


try:
    from primihub.FL.stats.mean_var import col_mean_local
    from primihub.FL.stats.min_max import col_min_local, col_max_local
    from primihub.FL.stats.sum import col_sum_local
    from primihub.FL.stats.norm import col_norm_local
    HAS_LOCAL = True
except (ImportError, AttributeError, ModuleNotFoundError):
    HAS_LOCAL = False


class TestFLStatsLocal:
    def test_local_mean(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = x.mean(axis=0)
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 3.0) < 1e-10

    def test_local_var(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = x.var(axis=0)
        assert abs(result[0] - 8.0/3.0) < 1e-10

    def test_local_min(self):
        x = np.array([[3.0, 8.0], [1.0, 6.0], [5.0, 2.0]])
        assert abs(x[:, 0].min() - 1.0) < 1e-10
        assert abs(x[:, 1].max() - 8.0) < 1e-10

    def test_local_sum(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(x[:, 0].sum() - 4.0) < 1e-10

    def test_local_norm(self):
        x = np.array([[3.0], [4.0]])
        n = np.linalg.norm(x)
        assert abs(n - 5.0) < 1e-10
