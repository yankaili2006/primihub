"""
Vertical Federated Linear Regression - Base Models
基础模型类，包含明文和CKKS加密版本
"""
import numpy as np


class LinearRegression_Host_Plaintext:
    """Host方的明文线性回归模型"""

    def __init__(self, x, learning_rate=0.2, alpha=0.0001):
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.weight = np.zeros(x.shape[1])
        self.bias = np.zeros(1)

    def compute_z(self, x, guest_z):
        """计算预测值，聚合各方的部分预测"""
        z = x.dot(self.weight) + self.bias
        z += np.array(guest_z).sum(axis=0)
        return z

    def compute_error(self, y, z):
        """计算误差"""
        return z - y

    def compute_grad(self, x, error):
        """计算梯度"""
        dw = x.T.dot(error) / x.shape[0] + self.alpha * self.weight
        db = error.mean(axis=0, keepdims=True)
        return dw, db

    def gradient_descent(self, x, error):
        """梯度下降更新参数"""
        dw, db = self.compute_grad(x, error)
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, x, error):
        """训练一步"""
        self.gradient_descent(x, error)


class LinearRegression_Host_CKKS(LinearRegression_Host_Plaintext):
    """Host方的CKKS加密线性回归模型"""

    def compute_enc_z(self, x, guest_z):
        """计算加密状态下的预测值"""
        z = self.weight.mm(x.T) + self.bias
        z += sum(guest_z)
        return z

    def gradient_descent(self, x, error):
        """CKKS加密状态下的梯度下降"""
        factor = -self.learning_rate / x.shape[0]
        self.bias += error.sum() * factor
        self.weight += error.mm(factor * x) \
            + (-self.learning_rate * self.alpha) * self.weight


class LinearRegression_Guest_Plaintext:
    """Guest方的明文线性回归模型"""

    def __init__(self, x, learning_rate=0.2, alpha=0.0001):
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.weight = np.zeros(x.shape[1])

    def compute_z(self, x):
        """计算本地特征的部分预测值"""
        return x.dot(self.weight)

    def compute_grad(self, x, error):
        """计算梯度"""
        dw = x.T.dot(error) / x.shape[0] + self.alpha * self.weight
        return dw

    def gradient_descent(self, x, error):
        """梯度下降更新参数"""
        dw = self.compute_grad(x, error)
        self.weight -= self.learning_rate * dw

    def fit(self, x, error):
        """训练一步"""
        self.gradient_descent(x, error)


class LinearRegression_Guest_CKKS(LinearRegression_Guest_Plaintext):
    """Guest方的CKKS加密线性回归模型"""

    def compute_enc_z(self, x):
        """计算加密状态下的部分预测值"""
        return self.weight.mm(x.T)

    def gradient_descent(self, x, error):
        """CKKS加密状态下的梯度下降"""
        factor = -self.learning_rate / x.shape[0]
        self.weight += error.mm(factor * x) + \
            (-self.learning_rate * self.alpha) * self.weight
