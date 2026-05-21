"""
Vertical Federated XGBoost - Base Models
基础模型类，包含树结构和梯度计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class TreeNode:
    """决策树节点"""

    def __init__(self,
                 node_id: int = 0,
                 depth: int = 0,
                 is_leaf: bool = False,
                 weight: float = 0.0):
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = is_leaf
        self.weight = weight

        # 分裂信息
        self.split_party: Optional[str] = None  # 'host' or 'guest'
        self.split_feature: Optional[str] = None
        self.split_value: Optional[float] = None
        self.split_gain: float = 0.0

        # 子节点
        self.left_child: Optional['TreeNode'] = None
        self.right_child: Optional['TreeNode'] = None

        # 样本索引
        self.sample_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            'node_id': self.node_id,
            'depth': self.depth,
            'is_leaf': self.is_leaf,
            'weight': self.weight,
            'split_party': self.split_party,
            'split_feature': self.split_feature,
            'split_value': self.split_value,
            'split_gain': self.split_gain,
        }
        if self.left_child:
            result['left_child'] = self.left_child.to_dict()
        if self.right_child:
            result['right_child'] = self.right_child.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        """从字典格式恢复"""
        node = cls(
            node_id=data['node_id'],
            depth=data['depth'],
            is_leaf=data['is_leaf'],
            weight=data['weight']
        )
        node.split_party = data.get('split_party')
        node.split_feature = data.get('split_feature')
        node.split_value = data.get('split_value')
        node.split_gain = data.get('split_gain', 0.0)

        if 'left_child' in data and data['left_child']:
            node.left_child = cls.from_dict(data['left_child'])
        if 'right_child' in data and data['right_child']:
            node.right_child = cls.from_dict(data['right_child'])
        return node


class XGBoostTree:
    """XGBoost决策树"""

    def __init__(self,
                 max_depth: int = 6,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 min_child_weight: float = 1.0,
                 min_child_sample: int = 1):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.min_child_sample = min_child_sample
        self.root: Optional[TreeNode] = None
        self.node_count = 0

    def compute_leaf_weight(self, g_sum: float, h_sum: float) -> float:
        """计算叶子节点权重"""
        return -g_sum / (h_sum + self.reg_lambda)

    def compute_split_gain(self,
                           g_left: float, h_left: float,
                           g_right: float, h_right: float) -> float:
        """计算分裂增益"""
        gain_left = g_left ** 2 / (h_left + self.reg_lambda)
        gain_right = g_right ** 2 / (h_right + self.reg_lambda)
        gain_parent = (g_left + g_right) ** 2 / (h_left + h_right + self.reg_lambda)
        return 0.5 * (gain_left + gain_right - gain_parent) - self.gamma

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'max_depth': self.max_depth,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'min_child_sample': self.min_child_sample,
            'root': self.root.to_dict() if self.root else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'XGBoostTree':
        """从字典格式恢复"""
        tree = cls(
            max_depth=data['max_depth'],
            reg_lambda=data['reg_lambda'],
            gamma=data['gamma'],
            min_child_weight=data['min_child_weight'],
            min_child_sample=data['min_child_sample']
        )
        if data.get('root'):
            tree.root = TreeNode.from_dict(data['root'])
        return tree


class XGBoostModel:
    """XGBoost模型"""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 min_child_weight: float = 1.0,
                 min_child_sample: int = 1,
                 objective: str = 'binary:logistic',
                 base_score: float = 0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.min_child_sample = min_child_sample
        self.objective = objective
        self.base_score = base_score

        self.trees: List[XGBoostTree] = []

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的sigmoid函数"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def compute_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度和Hessian"""
        if self.objective == 'binary:logistic':
            prob = self.sigmoid(y_pred)
            grad = prob - y_true
            hess = prob * (1 - prob)
        elif self.objective == 'reg:squarederror':
            grad = y_pred - y_true
            hess = np.ones_like(y_true)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")
        return grad, hess

    def predict_raw(self, predictions_per_tree: List[np.ndarray]) -> np.ndarray:
        """计算原始预测值"""
        y_pred = np.full(len(predictions_per_tree[0]) if predictions_per_tree else 0,
                         self.base_score)
        for tree_pred in predictions_per_tree:
            y_pred += self.learning_rate * tree_pred
        return y_pred

    def predict_proba(self, y_raw: np.ndarray) -> np.ndarray:
        """计算预测概率"""
        if self.objective == 'binary:logistic':
            return self.sigmoid(y_raw)
        return y_raw

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        """计算预测类别"""
        if self.objective == 'binary:logistic':
            return (self.predict_proba(y_raw) >= 0.5).astype(int)
        return y_raw

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'min_child_sample': self.min_child_sample,
            'objective': self.objective,
            'base_score': self.base_score,
            'trees': [tree.to_dict() for tree in self.trees]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'XGBoostModel':
        """从字典格式恢复"""
        model = cls(
            n_estimators=data['n_estimators'],
            max_depth=data['max_depth'],
            learning_rate=data['learning_rate'],
            reg_lambda=data['reg_lambda'],
            gamma=data['gamma'],
            min_child_weight=data['min_child_weight'],
            min_child_sample=data['min_child_sample'],
            objective=data['objective'],
            base_score=data['base_score']
        )
        model.trees = [XGBoostTree.from_dict(t) for t in data.get('trees', [])]
        return model


def find_best_split(x: np.ndarray,
                    g: np.ndarray,
                    h: np.ndarray,
                    reg_lambda: float = 1.0,
                    gamma: float = 0.0,
                    min_child_weight: float = 1.0,
                    min_child_sample: int = 1,
                    n_bins: int = 256) -> Tuple[Optional[float], float, float, float, float, float]:
    """
    寻找单个特征的最佳分裂点

    Returns:
        best_split_value, best_gain, g_left, h_left, g_right, h_right
    """
    n_samples = len(x)
    if n_samples < 2 * min_child_sample:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0

    # 获取分裂候选点
    unique_values = np.unique(x)
    if len(unique_values) <= 1:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0

    # 使用分位数作为候选分裂点
    if len(unique_values) > n_bins:
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        split_candidates = np.percentile(x, percentiles)
        split_candidates = np.unique(split_candidates)
    else:
        split_candidates = (unique_values[:-1] + unique_values[1:]) / 2

    g_total = g.sum()
    h_total = h.sum()

    best_gain = 0.0
    best_split = None
    best_g_left, best_h_left = 0.0, 0.0
    best_g_right, best_h_right = g_total, h_total

    for split_value in split_candidates:
        left_mask = x <= split_value
        n_left = left_mask.sum()
        n_right = n_samples - n_left

        if n_left < min_child_sample or n_right < min_child_sample:
            continue

        g_left = g[left_mask].sum()
        h_left = h[left_mask].sum()
        g_right = g_total - g_left
        h_right = h_total - h_left

        if h_left < min_child_weight or h_right < min_child_weight:
            continue

        # 计算增益
        gain_left = g_left ** 2 / (h_left + reg_lambda)
        gain_right = g_right ** 2 / (h_right + reg_lambda)
        gain_parent = g_total ** 2 / (h_total + reg_lambda)
        gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma

        if gain > best_gain:
            best_gain = gain
            best_split = split_value
            best_g_left, best_h_left = g_left, h_left
            best_g_right, best_h_right = g_right, h_right

    return best_split, best_gain, best_g_left, best_h_left, best_g_right, best_h_right


def find_best_split_from_histogram(hist_g: np.ndarray,
                                   hist_h: np.ndarray,
                                   hist_count: np.ndarray,
                                   bin_edges: np.ndarray,
                                   reg_lambda: float = 1.0,
                                   gamma: float = 0.0,
                                   min_child_weight: float = 1.0,
                                   min_child_sample: int = 1) -> Tuple[Optional[float], float, float, float, float, float]:
    """
    从直方图中寻找最佳分裂点（用于加密场景）

    Returns:
        best_split_value, best_gain, g_left, h_left, g_right, h_right
    """
    g_total = hist_g.sum()
    h_total = hist_h.sum()
    n_total = hist_count.sum()

    if n_total < 2 * min_child_sample:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0

    best_gain = 0.0
    best_split = None
    best_g_left, best_h_left = 0.0, 0.0
    best_g_right, best_h_right = g_total, h_total

    g_left_cumsum = np.cumsum(hist_g)
    h_left_cumsum = np.cumsum(hist_h)
    count_cumsum = np.cumsum(hist_count)

    for i in range(len(bin_edges) - 1):
        n_left = count_cumsum[i]
        n_right = n_total - n_left

        if n_left < min_child_sample or n_right < min_child_sample:
            continue

        g_left = g_left_cumsum[i]
        h_left = h_left_cumsum[i]
        g_right = g_total - g_left
        h_right = h_total - h_left

        if h_left < min_child_weight or h_right < min_child_weight:
            continue

        # 计算增益
        gain_left = g_left ** 2 / (h_left + reg_lambda)
        gain_right = g_right ** 2 / (h_right + reg_lambda)
        gain_parent = g_total ** 2 / (h_total + reg_lambda)
        gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma

        if gain > best_gain:
            best_gain = gain
            best_split = bin_edges[i + 1]
            best_g_left, best_h_left = g_left, h_left
            best_g_right, best_h_right = g_right, h_right

    return best_split, best_gain, best_g_left, best_h_left, best_g_right, best_h_right
