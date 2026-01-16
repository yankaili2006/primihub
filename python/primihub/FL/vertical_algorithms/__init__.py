# Vertical Federated Learning Algorithms
# This module provides unified interfaces for vertical federated learning algorithms

from .linear_regression import (
    VFLLinearRegressionHost,
    VFLLinearRegressionGuest,
    VFLLinearRegressionCoordinator,
)

from .logistic_regression import (
    VFLLogisticRegressionHost,
    VFLLogisticRegressionGuest,
    VFLLogisticRegressionCoordinator,
)

from .xgboost import (
    VFLXGBoostHost,
    VFLXGBoostGuest,
)

__all__ = [
    # Linear Regression
    'VFLLinearRegressionHost',
    'VFLLinearRegressionGuest',
    'VFLLinearRegressionCoordinator',
    # Logistic Regression
    'VFLLogisticRegressionHost',
    'VFLLogisticRegressionGuest',
    'VFLLogisticRegressionCoordinator',
    # XGBoost
    'VFLXGBoostHost',
    'VFLXGBoostGuest',
]
