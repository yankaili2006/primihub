"""
Federated Statistical Hypothesis Tests
"""

import numpy as np
from typing import Tuple, Optional, Union
from primihub.context import Context


class FederatedHypothesisTests:
    """Federated statistical hypothesis tests"""

    def __init__(self, protocol="ABY3"):
        self.protocol = protocol
        self._setup_mpc_executor()

    def _setup_mpc_executor(self):
        """Setup MPC executor for secure computations"""
        try:
            import ph_secure_lib as ph_slib

            cert_config = Context.cert_config
            root_ca_path = cert_config.get("root_ca_path", "")
            key_path = cert_config.get("key_path", "")
            cert_path = cert_config.get("cert_path", "")
            self.mpc_executor = ph_slib.MPCExecutor(
                Context.message, self.protocol, root_ca_path, key_path, cert_path
            )
        except ImportError:
            self.mpc_executor = None
            print("Warning: ph_secure_lib not available, using local computations only")

    def t_test(
        self, group1_data: np.ndarray, group2_data: np.ndarray, equal_var: bool = True
    ) -> Tuple[float, float, float]:
        """
        Perform federated two-sample t-test

        Args:
            group1_data: Data from group 1
            group2_data: Data from group 2
            equal_var: If True, assume equal population variances

        Returns:
            t_statistic: T-statistic
            df: Degrees of freedom
            p_value: Two-tailed p-value
        """
        if self.mpc_executor:
            # Use MPC for secure computation
            return self.mpc_executor.t_test(group1_data, group2_data, equal_var)
        else:
            # Fallback to local computation (for testing)
            return self._local_t_test(group1_data, group2_data, equal_var)

    def _local_t_test(
        self, group1: np.ndarray, group2: np.ndarray, equal_var: bool = True
    ) -> Tuple[float, float, float]:
        """Local implementation of t-test for testing"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        if equal_var:
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(var1 / n1 + var2 / n2)
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            )

        t_stat = (mean1 - mean2) / se

        # Two-tailed p-value using t-distribution
        from scipy import stats

        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return t_stat, df, p_value

    def f_test(
        self, group1_data: np.ndarray, group2_data: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Perform federated F-test for equality of variances

        Args:
            group1_data: Data from group 1
            group2_data: Data from group 2

        Returns:
            f_statistic: F-statistic
            df1: Degrees of freedom numerator
            df2: Degrees of freedom denominator
            p_value: P-value
        """
        if self.mpc_executor:
            return self.mpc_executor.f_test(group1_data, group2_data)
        else:
            return self._local_f_test(group1_data, group2_data)

    def _local_f_test(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Local implementation of F-test for testing"""
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        # F-statistic (larger variance / smaller variance)
        if var1 >= var2:
            f_stat = var1 / var2
            df1, df2 = n1 - 1, n2 - 1
        else:
            f_stat = var2 / var1
            df1, df2 = n2 - 1, n1 - 1

        # Two-tailed p-value
        from scipy import stats

        p_value = 2 * min(
            stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2)
        )

        return f_stat, df1, df2, p_value

    def chi_square_test(
        self, observed: np.ndarray, expected: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Perform federated Chi-square test

        Args:
            observed: Observed frequency counts
            expected: Expected frequency counts (if None, assumes uniform distribution)

        Returns:
            chi2_statistic: Chi-square statistic
            df: Degrees of freedom
            p_value: P-value
        """
        if self.mpc_executor:
            return self.mpc_executor.chi_square_test(observed, expected)
        else:
            return self._local_chi_square_test(observed, expected)

    def _local_chi_square_test(
        self, observed: np.ndarray, expected: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Local implementation of Chi-square test for testing"""
        observed = np.asarray(observed)

        if expected is None:
            # Assume uniform distribution
            expected = np.full_like(
                observed, np.sum(observed) / len(observed), dtype=float
            )
        else:
            expected = np.asarray(expected)

        # Chi-square statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

        # P-value
        from scipy import stats

        p_value = 1 - stats.chi2.cdf(chi2, df)

        return chi2, df, p_value

    def correlation(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute federated Pearson correlation coefficient

        Args:
            x_data: Data for variable X
            y_data: Data for variable Y

        Returns:
            correlation: Pearson correlation coefficient
            p_value: P-value for correlation test
        """
        if self.mpc_executor:
            return self.mpc_executor.correlation(x_data, y_data)
        else:
            return self._local_correlation(x_data, y_data)

    def _local_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Local implementation of correlation for testing"""
        from scipy import stats

        correlation, p_value = stats.pearsonr(x, y)
        return correlation, p_value

    def regression(
        self, X_data: np.ndarray, y_data: np.ndarray, method: str = "linear"
    ) -> dict:
        """
        Perform federated regression analysis

        Args:
            X_data: Independent variables (2D array)
            y_data: Dependent variable (1D array)
            method: Type of regression ('linear' or 'logistic')

        Returns:
            Dictionary with regression results
        """
        if self.mpc_executor:
            return self.mpc_executor.regression(X_data, y_data, method)
        else:
            return self._local_regression(X_data, y_data, method)

    def _local_regression(
        self, X: np.ndarray, y: np.ndarray, method: str = "linear"
    ) -> dict:
        """Local implementation of regression for testing"""
        if method == "linear":
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)
            coefficients = model.coef_
            intercept = model.intercept_
            r_squared = model.score(X, y)

            return {
                "coefficients": coefficients,
                "intercept": intercept,
                "r_squared": r_squared,
                "method": "linear",
            }
        elif method == "logistic":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]

            return {
                "coefficients": coefficients,
                "intercept": intercept,
                "method": "logistic",
            }
        else:
            raise ValueError(f"Unknown regression method: {method}")
