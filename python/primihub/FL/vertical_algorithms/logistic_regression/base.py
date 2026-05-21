"""
Vertical Federated Logistic Regression - Base Models
基础模型类，包含明文和CKKS加密版本，支持二分类和多分类
"""
import numpy as np
import tenseal as ts


class LogisticRegression_Host_Plaintext:
    """Host方的明文逻辑回归模型"""

    def __init__(self, x, y, learning_rate=0.2, alpha=0.0001):
        self.learning_rate = learning_rate
        self.alpha = alpha

        max_y = max(y)
        if max_y == 1:
            self.weight = np.zeros(x.shape[1])
            self.bias = np.zeros(1)
            self.multiclass = False
            self.output_dim = 1
        else:
            self.weight = np.zeros((x.shape[1], max_y + 1))
            self.bias = np.zeros((1, max_y + 1))
            self.multiclass = True
            self.output_dim = max_y + 1

    def sigmoid(self, x):
        """数值稳定的sigmoid函数"""
        def _positive_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def _negative_sigmoid(x):
            exp = np.exp(x)
            return exp / (exp + 1)

        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x, dtype=np.float64)
        result[positive] = _positive_sigmoid(x[positive])
        result[negative] = _negative_sigmoid(x[negative])
        return result

    def softmax(self, x):
        """数值稳定的softmax函数"""
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def compute_z(self, x, guest_z):
        """计算预测值，聚合各方的部分预测"""
        z = x.dot(self.weight) + self.bias
        z += np.array(guest_z).sum(axis=0)
        return z

    def predict_prob(self, z):
        """计算预测概率"""
        if self.multiclass:
            return self.softmax(z)
        else:
            return self.sigmoid(z)

    def compute_error(self, y, z):
        """计算误差"""
        if self.multiclass:
            error = self.predict_prob(z)
            idx = np.arange(len(y))
            error[idx, y] -= 1
        else:
            error = self.predict_prob(z) - y
        return error

    def compute_regular_loss(self, guest_regular_loss):
        """计算正则化损失"""
        return (0.5 * self.alpha) * (self.weight ** 2).sum() + guest_regular_loss

    def BCELoss(self, y, z, regular_loss):
        """二分类交叉熵损失"""
        return (np.maximum(z, 0.).sum() - y.dot(z) +
                np.log1p(np.exp(-np.abs(z))).sum()) / z.shape[0] + regular_loss

    def CELoss(self, y, z, regular_loss, eps=1e-20):
        """多分类交叉熵损失"""
        prob = self.predict_prob(z)
        return -np.sum(np.log(np.clip(prob[np.arange(len(y)), y], eps, 1.))) \
            / z.shape[0] + regular_loss

    def loss(self, y, z, regular_loss):
        """计算损失"""
        if self.multiclass:
            return self.CELoss(y, z, regular_loss)
        else:
            return self.BCELoss(y, z, regular_loss)

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


class LogisticRegression_Host_CKKS(LogisticRegression_Host_Plaintext):
    """Host方的CKKS加密逻辑回归模型"""

    def compute_enc_z(self, x, guest_z):
        """计算加密状态下的预测值"""
        z = self.weight.mm(x.T) + self.bias
        z += sum(guest_z)
        return z

    def compute_error(self, y, z):
        """计算加密状态下的近似误差"""
        if self.multiclass:
            error = z + 1 - self.output_dim * np.eye(self.output_dim)[y].T
        else:
            error = 2. + z - 4 * y
        return error

    def compute_regular_loss(self, guest_regular_loss):
        """计算正则化损失"""
        if self.multiclass and isinstance(self.weight, ts.CKKSTensor):
            return (0.5 * self.alpha) * (self.weight ** 2).sum().sum() \
                + guest_regular_loss
        else:
            return super().compute_regular_loss(guest_regular_loss)

    def BCELoss(self, y, z, regular_loss):
        """加密状态下的近似二分类损失"""
        return z.dot((0.5 - y) / y.shape[0]) + regular_loss

    def CELoss(self, y, z, regular_loss):
        """加密状态下的近似多分类损失"""
        factor = 1. / (y.shape[0] * self.output_dim)
        if isinstance(z, ts.CKKSTensor):
            return (z * factor
                    - z * ((np.eye(self.output_dim)[y].T
                            + np.random.normal(0, 1e-4, (self.output_dim, y.shape[0])))
                           * factor)).sum().sum() + regular_loss
        else:
            return np.sum(np.sum(z, axis=1) - z[np.arange(len(y)), y]) \
                * factor + regular_loss

    def loss(self, y, z, regular_loss):
        """计算损失"""
        if self.multiclass:
            return self.CELoss(y, z, regular_loss)
        else:
            return self.BCELoss(y, z, regular_loss)

    def gradient_descent(self, x, error):
        """CKKS加密状态下的梯度下降"""
        if self.multiclass:
            factor = -self.learning_rate / (self.output_dim * x.shape[0])
            self.bias += error.sum(axis=1).reshape((self.output_dim, 1)) * factor
        else:
            factor = -self.learning_rate / x.shape[0]
            self.bias += error.sum() * factor
        self.weight += error.mm(factor * x) \
            + (-self.learning_rate * self.alpha) * self.weight


class LogisticRegression_Guest_Plaintext:
    """Guest方的明文逻辑回归模型"""

    def __init__(self, x, learning_rate=0.2, alpha=0.0001, output_dim=1):
        self.learning_rate = learning_rate
        self.alpha = alpha

        if output_dim > 2:
            self.weight = np.zeros((x.shape[1], output_dim))
            self.multiclass = True
        else:
            self.weight = np.zeros(x.shape[1])
            self.multiclass = False

    def compute_z(self, x):
        """计算本地特征的部分预测值"""
        return x.dot(self.weight)

    def compute_regular_loss(self):
        """计算正则化损失"""
        return (0.5 * self.alpha) * (self.weight ** 2).sum()

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


class LogisticRegression_Guest_CKKS(LogisticRegression_Guest_Plaintext):
    """Guest方的CKKS加密逻辑回归模型"""

    def __init__(self, x, learning_rate=0.2, alpha=0.0001, output_dim=1):
        super().__init__(x, learning_rate, alpha, output_dim)
        self.output_dim = output_dim

    def compute_enc_z(self, x):
        """计算加密状态下的部分预测值"""
        return self.weight.mm(x.T)

    def compute_regular_loss(self):
        """计算正则化损失"""
        if self.multiclass and isinstance(self.weight, ts.CKKSTensor):
            return (0.5 * self.alpha) * (self.weight ** 2).sum().sum()
        else:
            return super().compute_regular_loss()

    def gradient_descent(self, x, error):
        """CKKS加密状态下的梯度下降"""
        if self.multiclass:
            factor = -self.learning_rate / (self.output_dim * x.shape[0])
        else:
            factor = -self.learning_rate / x.shape[0]
        self.weight += error.mm(factor * x) + \
            (-self.learning_rate * self.alpha) * self.weight
