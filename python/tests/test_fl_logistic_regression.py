import numpy as np
import pytest

np.float = float

from primihub.FL.logistic_regression.base import LogisticRegression, LogisticRegression_DPSGD
from primihub.FL.logistic_regression.vfl_base import (
    LogisticRegression_Guest_Plaintext,
    LogisticRegression_Guest_CKKS,
    LogisticRegression_Host_Plaintext,
    LogisticRegression_Host_CKKS,
)


@pytest.fixture
def binary_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = np.random.randint(0, 3, size=150)
    return X, y


class TestLogisticRegressionBase:
    def test_binary_init(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        assert not model.multiclass
        assert model.weight.shape == (X.shape[1],)
        assert model.bias.shape == (1,)

    def test_multiclass_init(self, multiclass_data):
        X, y = multiclass_data
        model = LogisticRegression(X, y)
        assert model.multiclass
        assert model.weight.shape == (X.shape[1], 3)
        assert model.bias.shape == (1, 3)

    def test_sigmoid(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        z = np.array([-10, -1, 0, 1, 10], dtype=np.float64)
        prob = model.sigmoid(z)
        assert np.all(prob >= 0) and np.all(prob <= 1)
        assert prob[0] < 1e-4
        assert prob[-1] > 1 - 1e-4
        assert prob[2] == 0.5

    def test_softmax(self, multiclass_data):
        X, y = multiclass_data
        model = LogisticRegression(X, y)
        z = np.array([[1, 2, 3], [0, 0, 0], [-1, 0, 1]], dtype=np.float64)
        prob = model.softmax(z)
        for row in prob:
            assert abs(row.sum() - 1) < 1e-6

    def test_predict_prob_binary(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        prob = model.predict_prob(X)
        assert prob.shape == (len(X),)
        assert np.all(prob >= 0) and np.all(prob <= 1)

    def test_predict_binary(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_gradient_descent_reduces_loss(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        loss_before = model.loss(X, y)
        model.fit(X, y)
        loss_after = model.loss(X, y)
        assert loss_after <= loss_before + 1e-10

    def test_score_improves_after_training(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        score_before = model.score(X, y)
        for _ in range(100):
            model.fit(X, y)
        score_after = model.score(X, y)
        assert score_after >= score_before

    def test_get_set_theta(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        theta = model.get_theta()
        model.set_theta(theta * 2)
        assert np.allclose(model.get_theta(), theta * 2)

    def test_bce_loss_shape(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(X, y)
        loss = model.BCELoss(X, y)
        assert isinstance(loss, float)

    def test_ce_loss_shape(self, multiclass_data):
        X, y = multiclass_data
        model = LogisticRegression(X, y)
        loss = model.CELoss(X, y)
        assert isinstance(loss, float)


class TestGuestPlaintext:
    def test_init_binary(self):
        x = np.random.randn(50, 3)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=1)
        assert not guest.multiclass
        assert guest.weight.shape == (3,)

    def test_init_multiclass(self):
        x = np.random.randn(50, 3)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=3)
        assert guest.multiclass
        assert guest.weight.shape == (3, 3)

    def test_compute_z_binary(self):
        x = np.random.randn(20, 4)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=1)
        z = guest.compute_z(x)
        assert z.shape == (20,)

    def test_compute_z_multiclass(self):
        x = np.random.randn(20, 4)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=3)
        z = guest.compute_z(x)
        assert z.shape == (20, 3)

    def test_compute_regular_loss(self):
        x = np.random.randn(20, 4)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=1)
        loss = guest.compute_regular_loss()
        assert loss >= 0

    def test_fit_updates_weight(self):
        x = np.random.randn(30, 5)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=1)
        weight_before = guest.weight.copy()
        error = np.random.randn(30)
        guest.fit(x, error)
        assert not np.allclose(guest.weight, weight_before)

    def test_compute_grad_shape(self):
        x = np.random.randn(30, 5)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.01, output_dim=1)
        error = np.random.randn(30)
        dw = guest.compute_grad(x, error)
        assert dw.shape == (5,)

    def test_multiple_fitting_steps(self):
        x = np.random.randn(50, 3)
        guest = LogisticRegression_Guest_Plaintext(x, learning_rate=0.1, alpha=0.001, output_dim=1)
        errors = [np.random.randn(50) for _ in range(10)]
        weights = []
        for e in errors:
            guest.fit(x, e)
            weights.append(guest.weight.copy())
        diffs = [np.linalg.norm(weights[i] - weights[i-1]) for i in range(1, len(weights))]
        assert all(d > 0 for d in diffs)


class TestHostPlaintext:
    def test_init_binary(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        assert not host.multiclass
        assert host.output_dim == 1

    def test_init_multiclass(self, multiclass_data):
        X, y = multiclass_data
        host = LogisticRegression_Host_Plaintext(X, y)
        assert host.multiclass
        assert host.output_dim == 3

    def test_compute_z(self, binary_data):
        X, y = binary_data
        X2 = np.random.randn(len(X), 2)
        host = LogisticRegression_Host_Plaintext(X, y)
        z = host.compute_z(X, np.random.randn(len(X)))
        assert z.shape == (len(X),)

    def test_predict_prob_binary(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        z = np.random.randn(len(X))
        prob = host.predict_prob(z)
        assert np.all(prob >= 0) and np.all(prob <= 1)

    def test_predict_prob_multiclass(self, multiclass_data):
        X, y = multiclass_data
        host = LogisticRegression_Host_Plaintext(X, y)
        z = np.random.randn(len(X), 3)
        prob = host.predict_prob(z)
        for row in prob:
            assert abs(row.sum() - 1) < 1e-6

    def test_compute_error(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        z = np.random.randn(len(y))
        error = host.compute_error(y, z)
        assert error.shape == (len(y),)

    def test_compute_grad_shapes(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        error = np.random.randn(len(y))
        dw, db = host.compute_grad(X, error)
        assert dw.shape == host.weight.shape
        assert db.shape == host.bias.shape

    def test_fit_updates_params(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        w_before = host.weight.copy()
        b_before = host.bias.copy()
        error = np.random.randn(len(y))
        host.fit(X, error)
        assert not np.allclose(host.weight, w_before) or not np.allclose(host.bias, b_before)

    def test_loss_decreases_after_fit(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        guest_z = host.compute_z(X, np.zeros(len(X)))
        regular_loss = 0.0
        loss_before = host.loss(y, guest_z, regular_loss)
        for _ in range(50):
            z = host.compute_z(X, np.zeros(len(X)))
            error = host.compute_error(y, z)
            host.fit(X, error)
        z_after = host.compute_z(X, np.zeros(len(X)))
        loss_after = host.loss(y, z_after, regular_loss)
        assert loss_after < loss_before

    def test_regular_loss_computation(self, binary_data):
        X, y = binary_data
        host = LogisticRegression_Host_Plaintext(X, y)
        guest_loss = 0.5
        reg = host.compute_regular_loss(guest_loss)
        assert reg > 0


class TestCKKSGuest:
    @pytest.mark.skip(reason="CKKS requires tenseal which may not be available")
    def test_placeholder(self):
        pass


class TestCKKSHost:
    @pytest.mark.skip(reason="CKKS requires tenseal which may not be available")
    def test_placeholder(self):
        pass


class TestDPSGD:
    def test_dpsgd_init(self, binary_data):
        X, y = binary_data
        model = LogisticRegression_DPSGD(X, y, learning_rate=0.1, noise_multiplier=1.0, l2_norm_clip=1.0)
        assert model.noise_multiplier == 1.0
        assert model.l2_norm_clip == 1.0

    def test_dpsgd_noise_shape(self, binary_data):
        X, y = binary_data
        model = LogisticRegression_DPSGD(X, y, learning_rate=0.1, noise_multiplier=1.0, l2_norm_clip=1.0)
        noisy = model.add_noise(np.zeros(10))
        assert noisy.shape == (10,)
        assert not np.allclose(noisy, 0)
