# Federated Statistical Tests

This document describes the implementation of federated statistical tests in PrimiHub.

## Overview

PrimiHub now supports federated statistical hypothesis tests that allow multiple parties to collaboratively perform statistical analyses without sharing their raw data. The tests are implemented using secure multi-party computation (MPC) techniques.

## Available Tests

### 1. T-Test
- **Purpose**: Compare means between two groups
- **Implementation**: `MPCTTest` class
- **Output**: t-statistic, degrees of freedom, p-value
- **Use Case**: Determine if there's a significant difference between two groups

### 2. F-Test
- **Purpose**: Compare variances between two groups
- **Implementation**: `MPCFTest` class
- **Output**: F-statistic, degrees of freedom (numerator, denominator), p-value
- **Use Case**: Test for equality of variances

### 3. Chi-Square Test
- **Purpose**: Test for independence or goodness of fit
- **Implementation**: `MPCChiSquareTest` class
- **Output**: Chi-square statistic, degrees of freedom, p-value
- **Use Case**: Categorical data analysis

### 4. Correlation Analysis
- **Purpose**: Measure linear relationship between two variables
- **Implementation**: `MPCCorrelation` class
- **Output**: Correlation coefficient, p-value
- **Use Case**: Relationship analysis

### 5. Regression Analysis
- **Purpose**: Model relationship between variables
- **Implementation**: Uses existing regression infrastructure
- **Output**: Coefficients, intercept, R-squared
- **Use Case**: Predictive modeling

## Architecture

### C++ Implementation

#### Core Classes
1. `MPCStatisticsOperator` - Base class for all statistical operations
2. `MPCTTest` - T-test implementation
3. `MPCFTest` - F-test implementation
4. `MPCChiSquareTest` - Chi-square test implementation
5. `MPCCorrelation` - Correlation analysis implementation

#### Key Files
- `src/primihub/executor/statistical_tests.h` - Header file
- `src/primihub/executor/statistical_tests.cc` - Implementation
- `src/primihub/executor/statistics.h` - Updated with new enum values
- `src/primihub/algorithm/mpc_statistics.cc` - Updated to support new tests

### Python Interface

#### Core Module
- `python/primihub/FL/stats/hypothesis_tests.py` - Python wrapper
- `python/primihub/MPC/statistics.py` - Extended MPC statistics interface

#### Key Classes
- `FederatedHypothesisTests` - Main Python interface
- `MPCJointStatistics` - Extended with new methods

## Protocol Buffers Updates

The following updates were made to `src/primihub/protos/common.proto`:

```protobuf
enum StatisticsOpType {
  MAX = 0;
  MIN = 1;
  AVG = 2;
  SUM = 3;
  T_TEST = 4;
  F_TEST = 5;
  CHI_SQUARE_TEST = 6;
  REGRESSION = 7;
  CORRELATION = 8;
}
```

## Usage Examples

### Python Usage

```python
from primihub.FL.stats import FederatedHypothesisTests

# Create test instance
test = FederatedHypothesisTests(protocol="ABY3")

# T-test example
t_stat, df, p_value = test.t_test(group1_data, group2_data)

# F-test example
f_stat, df1, df2, p_value = test.f_test(group1_data, group2_data)

# Chi-square test example
chi2, df, p_value = test.chi_square_test(observed_counts)

# Correlation example
correlation, p_value = test.correlation(x_data, y_data)

# Regression example
results = test.regression(X_data, y_data, method='linear')
```

### C++ Usage

```cpp
#include "src/primihub/executor/statistical_tests.h"

// Create T-test executor
auto t_test = std::make_unique<primihub::MPCTTest>();
t_test->setupChannel(party_id, comm_pkg);

// Run test
t_test->run(dataset, columns, col_dtype);

// Get results
eMatrix<double> results;
t_test->getResult(results);
```

## Security Considerations

1. **Data Privacy**: Raw data never leaves local parties
2. **Secure Computation**: Uses MPC protocols for all computations
3. **Result Privacy**: Only final statistics are revealed, not intermediate values
4. **Cryptographic Security**: Based on established MPC frameworks (ABY3)

## Performance Considerations

1. **Communication Overhead**: MPC requires multiple rounds of communication
2. **Computational Complexity**: Cryptographic operations add overhead
3. **Scalability**: Designed to work with large datasets across multiple parties
4. **Optimization**: Uses efficient MPC primitives for statistical computations

## Testing

Example test script: `example/code/statistical_tests_example.py`

Run the example:
```bash
cd /Users/primihub/github/primihub
python example/code/statistical_tests_example.py
```

## Future Enhancements

1. **Additional Tests**: ANOVA, Mann-Whitney U test, etc.
2. **Bayesian Methods**: Federated Bayesian inference
3. **Time Series Analysis**: Federated time series methods
4. **Non-parametric Tests**: Federated non-parametric alternatives
5. **Performance Optimizations**: More efficient MPC protocols

## References

1. PrimiHub Documentation
2. ABY3 MPC Framework
3. Statistical Methods for Federated Learning
4. Secure Multi-Party Computation for Statistics