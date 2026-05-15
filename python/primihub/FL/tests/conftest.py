"""FL test fixtures and mocks."""
import sys
import numpy as np
import pytest


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

    def send_selected(self, key, val, selected):
        for s in selected:
            self.store[f"{s}_{key}"] = val

    def recv_selected(self, key, selected):
        return [self.store.get(f"{s}_{key}") for s in selected]


@pytest.fixture
def mock_channel():
    return MockChannel()


@pytest.fixture
def regression_data():
    np.random.seed(42)
    x = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = x.dot(true_w) + 0.1 * np.random.randn(100)
    return x, y


@pytest.fixture
def binary_classification_data():
    np.random.seed(42)
    x = np.random.randn(100, 3)
    y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).astype(float)
    return x, y


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    n, dim, k = 150, 4, 3
    x = np.random.randn(n, dim)
    y = np.random.randint(0, k, n)
    return x, y
