import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from primihub.FL.linear_regression.base import (
        LinearRegression, LinearRegression_DPSGD, LinearRegression_Paillier
    )
    HAS_LINEAR = True
except ImportError as e:
    HAS_LINEAR = False

try:
    from primihub.FL.logistic_regression.base import (
        LogisticRegression, LogisticRegression_DPSGD
    )
    HAS_LOGISTIC = True
except (ImportError, AttributeError, ModuleNotFoundError):
    HAS_LOGISTIC = False

try:
    from primihub.FL.metrics import regression_metrics, classification_metrics
    HAS_METRICS = True
except (ImportError, AttributeError, ModuleNotFoundError):
    HAS_METRICS = False


@pytest.mark.skipif(not HAS_LINEAR, reason="linear_regression.base not importable")
class TestLinearRegressionModel:
    def test_plaintext_fit_predict(self, regression_data):
        x, y = regression_data
        model = LinearRegression(x, learning_rate=0.1, alpha=0.0001)
        for _ in range(200):
            model.fit(x, y)
        y_pred = model.predict(x)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.5, f"MSE too high: {mse}"

    def test_plaintext_coefficient_shape(self):
        x = np.random.randn(50, 3)
        y = x.dot([2.0, -1.0, 0.5])
        model = LinearRegression(x, learning_rate=0.1, alpha=0.0001)
        for _ in range(100):
            model.fit(x, y)
        assert model.weight.shape[0] == 3

    def test_dpsgd_training(self):
        np.random.seed(42)
        x = np.random.randn(80, 4)
        y = x.dot([1.0, -0.5, 2.0, -1.5]) + 0.05 * np.random.randn(80)
        model = LinearRegression_DPSGD(x, learning_rate=0.1, alpha=0.0001,
                                       noise_multiplier=0.5, l2_norm_clip=1.0,
                                       secure_mode=False)
        for _ in range(50):
            model.fit(x, y)
        y_pred = model.predict(x)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 2.0, f"DPSGD MSE too high: {mse}"

    def test_paillier_theta_encryption(self):
        x = np.random.randn(30, 2)
        y = x.dot([3.0, -2.0])
        model = LinearRegression_Paillier(x, learning_rate=0.1, alpha=0.0001)
        theta = model.get_theta()
        assert len(theta) == 3
        for _ in range(20):
            model.fit(x, y)

    def test_linear_regression_predict(self):
        x = np.random.randn(20, 4)
        y = x.dot([1.0, 2.0, 3.0, 4.0])
        model = LinearRegression(x, learning_rate=0.1, alpha=0.0001)
        for _ in range(100):
            model.fit(x, y)
        x_test = np.array([[1.0, 2.0, 3.0, 4.0]])
        pred = model.predict(x_test)
        expected = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0
        assert abs(pred[0] - expected) < 2.0

    def test_get_set_theta(self):
        x = np.random.randn(20, 3)
        model = LinearRegression(x, learning_rate=0.1, alpha=0.0001)
        theta1 = model.get_theta().copy()
        theta1[0] = 999
        model.set_theta(theta1)
        theta2 = model.get_theta()
        assert theta2[0] == 999

    def test_multiple_local_updates(self):
        x = np.random.randn(100, 5)
        true_w = np.array([2.0, -1.0, 0.5, 3.0, -1.5])
        y = x.dot(true_w)
        model = LinearRegression(x, learning_rate=0.2, alpha=0.0)
        losses = []
        for i in range(50):
            model.fit(x, y)
            losses.append(np.mean((y - model.predict(x)) ** 2))
        assert losses[-1] < losses[0], "Loss should decrease over time"

    def test_learning_rate_effect(self):
        x = np.random.randn(50, 2)
        y = x.dot([3.0, -2.0])
        model_fast = LinearRegression(x, learning_rate=1.0, alpha=0.0)
        model_slow = LinearRegression(x, learning_rate=0.001, alpha=0.0)
        for _ in range(30):
            model_fast.fit(x, y)
            model_slow.fit(x, y)
        fast_loss = np.mean((y - model_fast.predict(x)) ** 2)
        slow_loss = np.mean((y - model_slow.predict(x)) ** 2)
        assert fast_loss < slow_loss, "Higher LR should converge faster"

    def test_l2_regularization(self):
        x = np.random.randn(50, 10)
        y = x[:, 0] + 0.1 * np.random.randn(50)
        model_reg = LinearRegression(x, learning_rate=0.1, alpha=0.1)
        model_no_reg = LinearRegression(x, learning_rate=0.1, alpha=0.0)
        for _ in range(100):
            model_reg.fit(x, y)
            model_no_reg.fit(x, y)
        assert np.linalg.norm(model_reg.weight) < np.linalg.norm(model_no_reg.weight) + 1e-6


@pytest.mark.skipif(not HAS_LOGISTIC, reason="logistic_regression.base not importable")
class TestLogisticRegressionModel:
    def test_binary_classification(self, binary_classification_data):
        x, y = binary_classification_data
        model = LogisticRegression(x, y, learning_rate=0.5, alpha=0.0001)
        for _ in range(200):
            model.fit(x, y)
        y_pred = (model.predict(x) > 0.5).astype(float)
        acc = np.mean(y_pred == y)
        assert acc > 0.8, f"Accuracy too low: {acc}"

    def test_multiclass_classification(self):
        np.random.seed(42)
        n, dim, k = 150, 8, 3
        x = np.random.randn(n, dim)
        y = (x[:, 0] * 2 + x[:, 1] - x[:, 3] + np.random.randn(n) * 0.5)
        y = np.clip(np.round(y), 0, 2).astype(int)
        y = np.clip(y, 0, 2)
        model = LogisticRegression(x, y, learning_rate=0.3, alpha=0.0001)
        for _ in range(200):
            model.fit(x, y)
        y_pred = model.predict(x)
        acc = np.mean(y_pred == y)
        assert acc > 0.6, f"Multiclass accuracy too low: {acc}"

    def test_dpsgd_logistic(self):
        np.random.seed(42)
        x = np.random.randn(60, 2)
        y = (x[:, 0] - x[:, 1] > 0).astype(float)
        model = LogisticRegression_DPSGD(x, y, learning_rate=0.5, alpha=0.0001,
                                         noise_multiplier=0.8, l2_norm_clip=1.0,
                                         secure_mode=False)
        for _ in range(100):
            model.fit(x, y)
        y_pred = (model.predict(x) > 0.5).astype(float)
        acc = np.mean(y_pred == y)
        assert acc > 0.6, f"DPSGD accuracy too low: {acc}"


@pytest.mark.skipif(not HAS_METRICS, reason="metrics not importable")
class TestMetrics:
    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - mse / np.var(y_true)
        assert mae < 0.2
        assert r2 > 0.95

    def test_classification_accuracy(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        acc = np.mean(y_true == y_pred)
        assert acc == 0.8

    def test_precision_recall_f1(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        assert precision == 2/3
        assert recall == 2/3
        assert abs(f1 - 2/3) < 1e-10

    def test_regression_metrics_formulas(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ev = 1 - np.var(y_true - y_pred) / np.var(y_true)
        assert abs(mae - 0.5) < 1e-10
        assert mse == np.mean([0.25, 0.25, 0.25])
        assert abs(rmse - 0.5) < 1e-10


class TestVFLGuest:
    def test_guest_compute_z(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.5, 0.5])
        z = x.dot(w)
        assert list(z) == [1.5, 3.5]

    def test_guest_gradient(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        error = np.array([0.5, -0.5])
        dw = x.T.dot(error) / x.shape[0]
        expected = (1.0 * 0.5 + 3.0 * -0.5) / 2, (2.0 * 0.5 + 4.0 * -0.5) / 2
        assert abs(dw[0] - expected[0]) < 1e-10
        assert abs(dw[1] - expected[1]) < 1e-10


class TestVFLHost:
    def test_host_compute_z(self):
        x = np.array([[1.0, 2.0]])
        w = np.array([0.3, 0.7])
        b = np.array([0.1])
        guest_z = 1.5
        z = x.dot(w) + b + guest_z
        expected = 1.0*0.3 + 2.0*0.7 + 0.1 + 1.5
        assert abs(z[0] - expected) < 1e-10

    def test_host_gradient_descent(self):
        x = np.array([[1.0], [2.0], [3.0]])
        error = np.array([0.1, -0.2, 0.3])
        lr = 0.1
        w = np.array([0.5])
        dw = x.T.dot(error) / x.shape[0]
        w_new = w - lr * dw
        assert abs(w_new[0] - 0.5 + 0.1 * (0.1*1 - 0.2*2 + 0.3*3)/3) < 1e-10


class TestHFLClient:
    def test_client_weighted_average(self):
        w1 = np.array([1.0, 2.0])
        w2 = np.array([3.0, 4.0])
        n1, n2 = 100, 200
        avg = np.average([w1, w2], weights=[n1, n2], axis=0)
        expected = (w1 * n1 + w2 * n2) / (n1 + n2)
        np.testing.assert_array_almost_equal(avg, expected)

    def test_client_scaler(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x_scaled = (x - mean) / std
        assert abs(x_scaled.mean()) < 1e-10
        assert abs(x_scaled.std(axis=0)[0] - 1.0) < 1e-10
