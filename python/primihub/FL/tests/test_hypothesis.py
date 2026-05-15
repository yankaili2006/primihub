import numpy as np
from scipy import stats as scipy_stats
import random
import math


class TestHypothesisTests:
    def test_ttest_independent(self):
        np.random.seed(42)
        group1 = np.random.randn(30) + 1.0
        group2 = np.random.randn(30) + 0.0
        t_stat, p_val = scipy_stats.ttest_ind(group1, group2)
        assert t_stat > 2.0, "Groups should be significantly different"
        assert p_val < 0.05

    def test_ttest_no_difference(self):
        np.random.seed(42)
        group1 = np.random.randn(30)
        group2 = np.random.randn(30)
        t_stat, p_val = scipy_stats.ttest_ind(group1, group2)
        assert p_val > 0.01, "Groups should NOT be significantly different"

    def test_f_test(self):
        np.random.seed(42)
        group1 = np.random.randn(20) + 0.5
        group2 = np.random.randn(20) + 1.0
        group3 = np.random.randn(20) + 1.5
        f_stat, p_val = scipy_stats.f_oneway(group1, group2, group3)
        assert f_stat > 1.0
        assert p_val < 0.1

    def test_chi_square_independence(self):
        observed = np.array([[50, 30], [20, 50]])
        chi2, p_val, dof, expected = scipy_stats.chi2_contingency(observed)
        assert chi2 > 5.0
        assert p_val < 0.05

    def test_chi_square_no_association(self):
        np.random.seed(42)
        observed = np.random.randint(20, 40, size=(2, 3))
        chi2, p_val, dof, expected = scipy_stats.chi2_contingency(observed)
        assert p_val > 0.01

    def test_pearson_correlation(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = x * 0.8 + np.random.randn(50) * 0.6
        r, p_val = scipy_stats.pearsonr(x, y)
        assert abs(r) > 0.5
        assert p_val < 0.001

    def test_regression_significance(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2.0 * x + 1.0 + np.random.randn(100) * 0.5
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(x, y)
        assert abs(slope - 2.0) < 0.2
        assert p_val < 0.001

    def test_normal_distribution(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        stat, p_val = scipy_stats.normaltest(data)
        assert p_val > 0.01, "Data should be normally distributed"


class TestRandomSampling:
    def test_uniform_sampling(self):
        random.seed(42)
        samples = [random.random() for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert 0.45 < mean < 0.55

    def test_weighted_sampling(self):
        items = ['a', 'b', 'c']
        weights = [0.7, 0.2, 0.1]
        random.seed(42)
        counts = {'a': 0, 'b': 0, 'c': 0}
        for _ in range(10000):
            choice = random.choices(items, weights=weights, k=1)[0]
            counts[choice] += 1
        assert counts['a'] > counts['b'] > counts['c']

    def test_stratified_sampling(self):
        population = list(range(100))
        labels = [0] * 50 + [1] * 50
        sample_size = 20
        sampled_indices = []
        for label in [0, 1]:
            pool = [i for i in range(100) if labels[i] == label]
            sampled_indices.extend(random.sample(pool, sample_size // 2))
        sampled_labels = [labels[i] for i in sampled_indices]
        assert sum(sampled_labels) == 10


class TestMathUtils:
    def test_log1p_exp(self):
        x = np.array([-100, -10, -1, 0, 1, 10, 100])
        log1p_exp = np.log1p(np.exp(x))
        for val in log1p_exp:
            assert not np.isnan(val)
            assert not np.isinf(val)

    def test_softmax_stability(self):
        x = np.array([1000, 1010, 1005])
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        softmax = exp_x / exp_x.sum()
        assert abs(softmax.sum() - 1.0) < 1e-10
        assert np.argmax(softmax) == 1

    def test_sigmoid(self):
        x = np.array([-10, -1, 0, 1, 10])
        sigmoid = 1 / (1 + np.exp(-x))
        assert sigmoid[0] < 0.001
        assert abs(sigmoid[2] - 0.5) < 0.001
        assert sigmoid[4] > 0.999
