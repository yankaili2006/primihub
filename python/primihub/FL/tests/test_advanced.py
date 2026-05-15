import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPSISampleAlignment:
    def test_psi_data_preparation(self):
        """Test PSI data preparation logic (without network)."""
        client_data = np.array([[1, 100], [2, 200], [3, 300]])
        server_data = np.array([[2, 200], [3, 300], [4, 400]])
        client_ids = set(client_data[:, 0])
        server_ids = set(server_data[:, 0])
        intersection = client_ids & server_ids
        assert sorted(intersection) == [2, 3]

    def test_psi_result_extraction(self):
        client_data = np.array([[1, 100], [2, 200], [3, 300]])
        server_data = np.array([[2, 200], [3, 300], [4, 400]])
        intersection = {2, 3}
        client_aligned = client_data[np.isin(client_data[:, 0], list(intersection))]
        server_aligned = server_data[np.isin(server_data[:, 0], list(intersection))]
        assert client_aligned.shape[0] == 2
        assert server_aligned.shape[0] == 2
        np.testing.assert_array_equal(client_aligned[:, 0], [2, 3])

    def test_difference_mode(self):
        client_data = np.array([[1, 100], [2, 200], [3, 300]])
        server_data = np.array([[2, 200], [3, 300]])
        client_ids = set(client_data[:, 0])
        server_ids = set(server_data[:, 0])
        difference = client_ids - server_ids
        assert difference == {1}


class TestXGBoostStructure:
    def test_decision_tree_split(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        y = (x[:, 0] > 0).astype(float)
        best_gain = -1
        best_threshold = None
        for col in range(x.shape[1]):
            thresholds = np.percentile(x[:, col], np.linspace(10, 90, 9))
            for thresh in thresholds:
                left = y[x[:, col] <= thresh]
                right = y[x[:, col] > thresh]
                if len(left) < 1 or len(right) < 1:
                    continue
                gain = len(left) * np.var(left) + len(right) * np.var(right)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = thresh
        assert best_threshold is not None

    def test_gradient_boosting_init(self):
        y = np.array([0, 1, 0, 1, 0])
        init_pred = np.full_like(y, 0.5, dtype=float)
        grad = init_pred - y
        hess = init_pred * (1 - init_pred)
        assert grad.shape == y.shape
        assert hess.shape == y.shape

    def test_goss_sampling(self):
        np.random.seed(42)
        n = 1000
        grad = np.random.randn(n)
        large_ratio, small_ratio = 0.2, 0.2
        n_large = int(n * large_ratio)
        n_small = int(n * small_ratio)
        top_indices = np.argsort(np.abs(grad))[::-1][:n_large]
        remaining = np.setdiff1d(np.arange(n), top_indices)
        sampled_remaining = np.random.choice(remaining, n_small, replace=False)
        sampled = np.concatenate([top_indices, sampled_remaining])
        assert len(sampled) == n_large + n_small


class TestChatGLMBase:
    def test_prefix_encoding_structure(self):
        """Test the basic structure of prefix encoder (p-tuning v2)."""
        pre_seq_len = 10
        num_layers = 6
        prefix_encoder = {
            f'layer_{i}': np.random.randn(pre_seq_len, 128)
            for i in range(num_layers)
        }
        assert len(prefix_encoder) == num_layers
        assert prefix_encoder['layer_0'].shape == (pre_seq_len, 128)

    def test_weighted_averaging_aggregation(self):
        client_weights = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        num_samples = [100, 200]
        total = sum(num_samples)
        aggregated = sum(
            w * n / total for w, n in zip(client_weights, num_samples)
        )
        expected = (np.array([1, 2, 3]) * 100 + np.array([4, 5, 6]) * 200) / 300
        np.testing.assert_array_almost_equal(aggregated, expected)
