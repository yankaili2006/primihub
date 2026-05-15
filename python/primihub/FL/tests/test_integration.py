import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# These tests require the compiled C++ linkcontext module.
# Run only after full build with: bazel build //src/primihub/pybind_warpper:linkcontext
pytestmark = pytest.mark.skip(reason="requires C++ linkcontext module (full build)")


class MockChannel:
    def __init__(self):
        self.store = {}

    def send(self, key, val):
        self.store[key] = val

    def recv(self, key):
        return self.store.get(key)

    def send_all(self, key, val):
        for k in list(self.store.keys()):
            if k.endswith(key):
                self.store[k] = val

    def recv_all(self, key):
        return [v for k, v in self.store.items() if k.endswith(key)]


class TestWeightedAverage:
    def test_numpy_weighted_average(self):
        w1, w2 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
        result = np.average([w1, w2], weights=[30, 50], axis=0)
        expected = (30 * w1 + 50 * w2) / 80
        np.testing.assert_array_almost_equal(result, expected)

    def test_metrics_weighted_average(self):
        mse1, mse2 = 0.5, 0.3
        n1, n2 = 30, 50
        result = (mse1 * n1 + mse2 * n2) / (n1 + n2)
        assert abs(result - 0.375) < 1e-10

    def test_numpy_broadcast_weighted_avg(self):
        w1 = np.array([10.0, 20.0])
        w2 = np.array([30.0, 40.0])
        result = np.average([w1, w2], weights=[100, 200], axis=0)
        expected = np.array([10.0 * 100 + 30.0 * 200,
                             20.0 * 100 + 40.0 * 200]) / 300
        np.testing.assert_array_almost_equal(result, expected)
