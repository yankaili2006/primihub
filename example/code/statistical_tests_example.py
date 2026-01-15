#!/usr/bin/env python3
"""
Example script demonstrating federated statistical tests
"""

import numpy as np
from primihub.FL.stats import FederatedHypothesisTests


def example_t_test():
    """Example of federated T-test"""
    print("=== Federated T-Test Example ===")

    # Simulate data from two groups
    np.random.seed(42)
    group1 = np.random.normal(loc=10, scale=2, size=100)
    group2 = np.random.normal(loc=12, scale=2, size=100)

    # Create federated test instance
    test = FederatedHypothesisTests()

    # Perform t-test
    t_stat, df, p_value = test.t_test(group1, group2)

    print(f"Group 1: mean={np.mean(group1):.2f}, std={np.std(group1):.2f}")
    print(f"Group 2: mean={np.mean(group2):.2f}, std={np.std(group2):.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df:.1f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference")
    print()


def example_f_test():
    """Example of federated F-test"""
    print("=== Federated F-Test Example ===")

    # Simulate data with different variances
    np.random.seed(42)
    group1 = np.random.normal(loc=10, scale=2, size=100)
    group2 = np.random.normal(loc=10, scale=3, size=100)

    test = FederatedHypothesisTests()

    # Perform F-test
    f_stat, df1, df2, p_value = test.f_test(group1, group2)

    print(f"Group 1: variance={np.var(group1):.2f}")
    print(f"Group 2: variance={np.var(group2):.2f}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"Degrees of freedom: {df1}, {df2}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Variances are significantly different (p < 0.05)")
    else:
        print("Result: No significant difference in variances")
    print()


def example_chi_square_test():
    """Example of federated Chi-square test"""
    print("=== Federated Chi-Square Test Example ===")

    # Simulate categorical data
    observed = np.array([30, 20, 25, 25])  # Observed counts
    expected = np.array([25, 25, 25, 25])  # Expected uniform distribution

    test = FederatedHypothesisTests()

    # Perform Chi-square test
    chi2, df, p_value = test.chi_square_test(observed, expected)

    print(f"Observed counts: {observed}")
    print(f"Expected counts: {expected}")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Distribution differs from expected (p < 0.05)")
    else:
        print("Result: Distribution matches expected")
    print()


def example_correlation():
    """Example of federated correlation"""
    print("=== Federated Correlation Example ===")

    # Simulate correlated data
    np.random.seed(42)
    x = np.random.normal(size=100)
    y = 2 * x + np.random.normal(scale=0.5, size=100)  # y = 2x + noise

    test = FederatedHypothesisTests()

    # Compute correlation
    correlation, p_value = test.correlation(x, y)

    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Significant correlation (p < 0.05)")
    else:
        print("Result: No significant correlation")
    print()


def example_regression():
    """Example of federated regression"""
    print("=== Federated Regression Example ===")

    # Simulate regression data
    np.random.seed(42)
    n_samples = 100
    X = np.random.normal(size=(n_samples, 3))
    true_coeffs = np.array([1.5, -2.0, 0.5])
    y = X @ true_coeffs + np.random.normal(scale=0.5, size=n_samples)

    test = FederatedHypothesisTests()

    # Perform linear regression
    results = test.regression(X, y, method="linear")

    print("True coefficients:", true_coeffs)
    print("Estimated coefficients:", results["coefficients"])
    print(f"Intercept: {results['intercept']:.4f}")
    print(f"R-squared: {results['r_squared']:.4f}")
    print()


def main():
    """Run all examples"""
    print("Federated Statistical Tests Examples")
    print("=" * 40)

    example_t_test()
    example_f_test()
    example_chi_square_test()
    example_correlation()
    example_regression()

    print("All examples completed!")


if __name__ == "__main__":
    main()
