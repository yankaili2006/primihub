"""
Vertical Federated XGBoost - Host
Host方持有标签数据，负责计算梯度/Hessian并协调树的构建过程
"""
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import (
    save_json_file,
    save_pickle_file,
    load_pickle_file,
    save_csv_file
)
from primihub.FL.utils.dataset import read_data
from primihub.utils.logger_util import logger
from primihub.FL.psi import sample_alignment
from primihub.FL.metrics import classification_metrics, regression_metrics

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_array
from typing import Dict, List, Optional, Tuple, Any

from .base import XGBoostModel, XGBoostTree, TreeNode, find_best_split


class VFLXGBoostHost(BaseModel):
    """纵向联邦XGBoost - Host方

    Host方持有标签数据，负责：
    1. 计算梯度和Hessian
    2. 协调树的构建过程
    3. 收集各方的分裂候选并选择最优分裂
    4. 广播分裂决策
    5. 保存模型和评估指标
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        process = self.common_params['process']
        logger.info(f"process: {process}")
        if process == 'train':
            self.train()
        elif process == 'predict':
            self.predict()
        else:
            error_msg = f"Unsupported process: {process}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def train(self):
        # setup communication channels
        guest_channel = MultiGrpcClients(
            local_party=self.role_params['self_name'],
            remote_parties=self.roles['guest'],
            node_info=self.node_info,
            task_info=self.task_info
        )

        # load dataset
        selected_column = self.role_params['selected_column']
        x = read_data(
            data_info=self.role_params['data'],
            selected_column=selected_column
        )

        # psi - 隐私集合求交
        id_col = self.role_params.get("id")
        psi_protocol = self.common_params.get("psi")
        if isinstance(psi_protocol, str):
            x = sample_alignment(x, id_col, self.roles, psi_protocol)

        x = x.drop(id_col, axis=1)
        label = self.role_params['label']
        y = x.pop(label).values

        # 获取特征名
        feature_names = list(x.columns)
        x = check_array(x, dtype='numeric')

        # 获取XGBoost参数
        n_estimators = self.common_params.get('n_estimators', 100)
        max_depth = self.common_params.get('max_depth', 6)
        learning_rate = self.common_params.get('learning_rate', 0.1)
        reg_lambda = self.common_params.get('reg_lambda', 1.0)
        gamma = self.common_params.get('gamma', 0.0)
        min_child_weight = self.common_params.get('min_child_weight', 1.0)
        min_child_sample = self.common_params.get('min_child_sample', 1)
        objective = self.common_params.get('objective', 'binary:logistic')

        # 创建Host训练器
        host = PlaintextHost(
            x=x,
            y=y,
            feature_names=feature_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            min_child_sample=min_child_sample,
            objective=objective,
            guest_channel=guest_channel
        )

        # 开始训练
        logger.info("-------- start training --------")
        host.train()
        logger.info("-------- finish training --------")

        # 计算最终指标
        train_metrics = host.compute_final_metrics()
        save_json_file(train_metrics, self.role_params['metric_path'])

        # 保存模型
        model_file = {
            "selected_column": selected_column,
            "id": id_col,
            "label": label,
            "feature_names": feature_names,
            "model": host.model,
            "objective": objective
        }
        save_pickle_file(model_file, self.role_params['model_path'])

    def predict(self):
        # setup communication channels
        remote_parties = self.roles[self.role_params['others_role']]
        guest_channel = MultiGrpcClients(
            local_party=self.role_params['self_name'],
            remote_parties=remote_parties,
            node_info=self.node_info,
            task_info=self.task_info
        )

        # load model for prediction
        model_file = load_pickle_file(self.role_params['model_path'])

        # load dataset
        origin_data = read_data(data_info=self.role_params['data'])

        x = origin_data.copy()
        selected_column = model_file['selected_column']
        if selected_column:
            x = x[selected_column]
        id_col = model_file['id']
        psi_protocol = self.common_params.get("psi")
        if isinstance(psi_protocol, str):
            x = sample_alignment(x, id_col, self.roles, psi_protocol)
        if id_col in x.columns:
            x.pop(id_col)
        label = model_file['label']
        if label in x.columns:
            y = x.pop(label).values
        else:
            y = None

        feature_names = model_file['feature_names']
        x = check_array(x, dtype='numeric')

        # 预测
        model = model_file['model']
        objective = model_file['objective']

        # 创建预测器
        predictor = PlaintextPredictor(
            x=x,
            feature_names=feature_names,
            model=model,
            guest_channel=guest_channel
        )

        y_raw, y_pred = predictor.predict()

        if objective == 'binary:logistic':
            result = pd.DataFrame({
                'pred_prob': model.predict_proba(y_raw),
                'pred_y': y_pred
            })
        else:
            result = pd.DataFrame({
                'pred_y': y_pred
            })

        data_result = pd.concat([origin_data, result], axis=1)
        save_csv_file(data_result, self.role_params['predict_path'])


class PlaintextHost:
    """明文训练的Host实现"""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 feature_names: List[str],
                 n_estimators: int,
                 max_depth: int,
                 learning_rate: float,
                 reg_lambda: float,
                 gamma: float,
                 min_child_weight: float,
                 min_child_sample: int,
                 objective: str,
                 guest_channel: MultiGrpcClients):
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.guest_channel = guest_channel

        self.model = XGBoostModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            min_child_sample=min_child_sample,
            objective=objective
        )

        self.n_samples = x.shape[0]

        # 发送配置给Guest
        config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'reg_lambda': reg_lambda,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'min_child_sample': min_child_sample,
            'n_samples': self.n_samples
        }
        self.guest_channel.send_all('config', config)

    def train(self):
        """训练XGBoost模型"""
        n_samples = self.n_samples
        y_pred = np.full(n_samples, self.model.base_score)

        for tree_idx in range(self.model.n_estimators):
            logger.info(f"-------- Building tree {tree_idx + 1}/{self.model.n_estimators} --------")

            # 通知Guest开始新的树构建
            self.guest_channel.send_all('new_tree', {'tree_idx': tree_idx})

            # 计算梯度和Hessian
            grad, hess = self.model.compute_grad_hess(self.y, y_pred)

            # 构建树
            tree = self._build_tree(grad, hess)
            self.model.trees.append(tree)

            # 通知Guest当前树构建完成
            self.guest_channel.send_all('tree_done', True)

            # 更新预测值
            tree_pred = self._predict_tree_training(tree)
            y_pred += self.model.learning_rate * tree_pred

            # 计算并打印损失
            if self.model.objective == 'binary:logistic':
                prob = self.model.sigmoid(y_pred)
                loss = -np.mean(self.y * np.log(prob + 1e-10) + (1 - self.y) * np.log(1 - prob + 1e-10))
            else:
                loss = np.mean((y_pred - self.y) ** 2)
            logger.info(f"Tree {tree_idx + 1} - Loss: {loss:.6f}")

        # 通知Guest训练结束
        self.guest_channel.send_all('training_complete', True)

    def _build_tree(self, grad: np.ndarray, hess: np.ndarray) -> XGBoostTree:
        """构建一棵树"""
        tree = XGBoostTree(
            max_depth=self.model.max_depth,
            reg_lambda=self.model.reg_lambda,
            gamma=self.model.gamma,
            min_child_weight=self.model.min_child_weight,
            min_child_sample=self.model.min_child_sample
        )

        # 创建根节点
        sample_indices = list(range(self.n_samples))
        root = TreeNode(node_id=0, depth=0)
        root.sample_indices = sample_indices
        tree.node_count = 1

        # 使用队列进行广度优先构建
        node_queue = [root]

        while node_queue:
            node = node_queue.pop(0)

            # 发送节点处理请求
            self.guest_channel.send_all('process_node', {
                'node_id': node.node_id,
                'sample_indices': node.sample_indices,
                'depth': node.depth
            })

            if node.depth >= tree.max_depth or len(node.sample_indices) < 2 * tree.min_child_sample:
                # 标记为叶子节点
                g_sum = grad[node.sample_indices].sum()
                h_sum = hess[node.sample_indices].sum()
                node.is_leaf = True
                node.weight = tree.compute_leaf_weight(g_sum, h_sum)
                self.guest_channel.send_all('node_result', {'is_leaf': True})
                continue

            # 发送当前节点的梯度和Hessian
            node_grad = grad[node.sample_indices]
            node_hess = hess[node.sample_indices]
            self.guest_channel.send_all('grad_hess', {
                'grad': node_grad.tolist(),
                'hess': node_hess.tolist()
            })

            # Host方寻找本地最优分裂
            host_best_split = self._find_best_local_split(
                node.sample_indices, node_grad, node_hess
            )

            # 收集Guest的最优分裂候选
            guest_splits = self.guest_channel.recv_all('best_split')

            # 选择全局最优分裂
            best_split = host_best_split
            best_party = 'host'
            best_party_idx = -1

            for i, guest_split in enumerate(guest_splits):
                if guest_split['gain'] > best_split['gain']:
                    best_split = guest_split
                    best_party = 'guest'
                    best_party_idx = i

            if best_split['gain'] <= 0:
                # 无法分裂，标记为叶子节点
                g_sum = grad[node.sample_indices].sum()
                h_sum = hess[node.sample_indices].sum()
                node.is_leaf = True
                node.weight = tree.compute_leaf_weight(g_sum, h_sum)
                self.guest_channel.send_all('node_result', {'is_leaf': True})
                continue

            # 广播分裂决策
            if best_party == 'host':
                # Host方的特征被选中
                split_info = {
                    'is_leaf': False,
                    'split_party': 'host',
                    'split_feature': best_split['feature_name'],
                    'split_value': best_split['split_value'],
                    'party_idx': -1
                }
                self.guest_channel.send_all('node_result', split_info)

                # 计算左右子节点的样本索引
                feature_idx = best_split['feature_idx']
                left_indices = [idx for idx in node.sample_indices
                               if self.x[idx, feature_idx] <= best_split['split_value']]
                right_indices = [idx for idx in node.sample_indices
                                if self.x[idx, feature_idx] > best_split['split_value']]
            else:
                # Guest方的特征被选中
                split_info = {
                    'is_leaf': False,
                    'split_party': 'guest',
                    'split_feature': best_split['feature_name'],
                    'split_value': best_split['split_value'],
                    'party_idx': best_party_idx
                }
                self.guest_channel.send_all('node_result', split_info)

                # 接收Guest方计算的左右子节点样本索引
                indices_result = self.guest_channel.recv_all('split_indices')
                left_indices = indices_result[best_party_idx]['left_indices']
                right_indices = indices_result[best_party_idx]['right_indices']

            # 记录分裂信息
            node.split_party = best_party
            node.split_feature = best_split['feature_name']
            node.split_value = best_split['split_value']
            node.split_gain = best_split['gain']

            # 创建子节点
            if left_indices:
                left_child = TreeNode(
                    node_id=tree.node_count,
                    depth=node.depth + 1
                )
                left_child.sample_indices = left_indices
                tree.node_count += 1
                node.left_child = left_child
                node_queue.append(left_child)
            else:
                left_child = TreeNode(
                    node_id=tree.node_count,
                    depth=node.depth + 1,
                    is_leaf=True,
                    weight=0.0
                )
                tree.node_count += 1
                node.left_child = left_child

            if right_indices:
                right_child = TreeNode(
                    node_id=tree.node_count,
                    depth=node.depth + 1
                )
                right_child.sample_indices = right_indices
                tree.node_count += 1
                node.right_child = right_child
                node_queue.append(right_child)
            else:
                right_child = TreeNode(
                    node_id=tree.node_count,
                    depth=node.depth + 1,
                    is_leaf=True,
                    weight=0.0
                )
                tree.node_count += 1
                node.right_child = right_child

        tree.root = root
        return tree

    def _find_best_local_split(self,
                               sample_indices: List[int],
                               grad: np.ndarray,
                               hess: np.ndarray) -> Dict:
        """在Host的本地特征中寻找最优分裂"""
        best_split = {
            'feature_idx': None,
            'feature_name': None,
            'split_value': None,
            'gain': 0.0
        }

        for feature_idx in range(self.x.shape[1]):
            x_feature = self.x[sample_indices, feature_idx]

            split_value, gain, g_left, h_left, g_right, h_right = find_best_split(
                x_feature, grad, hess,
                self.model.reg_lambda,
                self.model.gamma,
                self.model.min_child_weight,
                self.model.min_child_sample
            )

            if gain > best_split['gain']:
                best_split = {
                    'feature_idx': feature_idx,
                    'feature_name': self.feature_names[feature_idx],
                    'split_value': split_value,
                    'gain': gain
                }

        return best_split

    def _predict_tree_training(self, tree: XGBoostTree) -> np.ndarray:
        """训练阶段使用样本索引快速预测"""
        predictions = np.zeros(self.n_samples)

        def traverse(node: TreeNode):
            if node.is_leaf:
                if node.sample_indices:
                    for idx in node.sample_indices:
                        predictions[idx] = node.weight
            else:
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)

        traverse(tree.root)
        return predictions

    def compute_final_metrics(self) -> Dict:
        """计算最终的评估指标"""
        # 使用训练阶段的快速预测
        y_pred_raw = np.full(self.n_samples, self.model.base_score)
        for tree in self.model.trees:
            tree_pred = self._predict_tree_training(tree)
            y_pred_raw += self.model.learning_rate * tree_pred

        if self.model.objective == 'binary:logistic':
            y_prob = self.model.predict_proba(y_pred_raw)
            metrics = classification_metrics(
                self.y,
                y_prob,
                multiclass=False,
                prefix="train_",
                metircs_name=[
                    "acc", "f1", "precision", "recall", "auc"
                ],
            )
        else:
            metrics = regression_metrics(
                self.y,
                y_pred_raw,
                prefix="train_",
                metircs_name=[
                    "ev", "maxe", "mae", "mse", "rmse", "medae", "r2"
                ],
            )

        return metrics


class PlaintextPredictor:
    """明文预测的Host实现"""

    def __init__(self,
                 x: np.ndarray,
                 feature_names: List[str],
                 model: XGBoostModel,
                 guest_channel: MultiGrpcClients):
        self.x = x
        self.feature_names = feature_names
        self.model = model
        self.guest_channel = guest_channel
        self.n_samples = x.shape[0]

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """执行预测"""
        # 发送预测配置
        self.guest_channel.send_all('predict_config', {
            'n_samples': self.n_samples,
            'n_trees': len(self.model.trees)
        })

        y_pred_raw = np.full(self.n_samples, self.model.base_score)

        for tree_idx, tree in enumerate(self.model.trees):
            # 通知开始预测当前树
            self.guest_channel.send_all('predict_tree', {'tree_idx': tree_idx})
            tree_pred = self._predict_tree(tree)
            y_pred_raw += self.model.learning_rate * tree_pred
            # 通知当前树预测完成
            self.guest_channel.send_all('tree_predict_done', True)

        # 通知预测结束
        self.guest_channel.send_all('prediction_complete', True)

        y_pred = self.model.predict(y_pred_raw)
        return y_pred_raw, y_pred

    def _predict_tree(self, tree: XGBoostTree) -> np.ndarray:
        """使用单棵树进行预测"""
        predictions = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            node = tree.root
            while node and not node.is_leaf:
                if node.split_party == 'host':
                    if node.split_feature in self.feature_names:
                        feature_idx = self.feature_names.index(node.split_feature)
                        if self.x[i, feature_idx] <= node.split_value:
                            node = node.left_child
                        else:
                            node = node.right_child
                    else:
                        # 特征在Guest方，需要查询
                        self.guest_channel.send_all('query_split', {
                            'sample_idx': i,
                            'split_feature': node.split_feature,
                            'split_value': node.split_value
                        })
                        go_left_results = self.guest_channel.recv_all('go_left')
                        go_left = any(r.get('go_left', False) for r in go_left_results if r.get('has_feature', False))
                        node = node.left_child if go_left else node.right_child
                else:
                    # Guest方的特征
                    self.guest_channel.send_all('query_split', {
                        'sample_idx': i,
                        'split_feature': node.split_feature,
                        'split_value': node.split_value
                    })
                    go_left_results = self.guest_channel.recv_all('go_left')
                    go_left = any(r.get('go_left', False) for r in go_left_results if r.get('has_feature', False))
                    node = node.left_child if go_left else node.right_child

            if node:
                predictions[i] = node.weight

        return predictions
