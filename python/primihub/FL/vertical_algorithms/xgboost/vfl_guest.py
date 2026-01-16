"""
Vertical Federated XGBoost - Guest
Guest方持有特征数据，参与联邦训练
"""
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, load_pickle_file
from primihub.FL.utils.dataset import read_data
from primihub.utils.logger_util import logger
from primihub.FL.psi import sample_alignment

from sklearn.utils.validation import check_array
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import find_best_split


class VFLXGBoostGuest(BaseModel):
    """纵向联邦XGBoost - Guest方

    Guest方持有特征数据，负责：
    1. 接收Host方的梯度和Hessian
    2. 在本地特征上寻找最优分裂点
    3. 发送分裂候选给Host
    4. 根据分裂决策计算样本划分
    5. 保存本地模型信息
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
        host_channel = GrpcClient(
            local_party=self.role_params['self_name'],
            remote_party=self.roles['host'],
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

        # 获取特征名
        feature_names = list(x.columns)
        x = check_array(x, dtype='numeric')

        # 创建Guest训练器
        guest = PlaintextGuest(
            x=x,
            feature_names=feature_names,
            host_channel=host_channel
        )

        # 开始训练
        logger.info("-------- start training --------")
        guest.train()
        logger.info("-------- finish training --------")

        # 保存模型信息
        model_file = {
            "selected_column": selected_column,
            "id": id_col,
            "feature_names": feature_names
        }
        save_pickle_file(model_file, self.role_params['model_path'])

    def predict(self):
        # setup communication channels
        remote_party = self.roles[self.role_params['others_role']]
        host_channel = GrpcClient(
            local_party=self.role_params['self_name'],
            remote_party=remote_party,
            node_info=self.node_info,
            task_info=self.task_info
        )

        # load model for prediction
        model_file = load_pickle_file(self.role_params['model_path'])

        # load dataset
        x = read_data(data_info=self.role_params['data'])

        selected_column = model_file['selected_column']
        if selected_column:
            x = x[selected_column]
        id_col = model_file['id']
        psi_protocol = self.common_params.get("psi")
        if isinstance(psi_protocol, str):
            x = sample_alignment(x, id_col, self.roles, psi_protocol)
        if id_col in x.columns:
            x.pop(id_col)

        feature_names = model_file['feature_names']
        x = check_array(x, dtype='numeric')

        # 创建预测器
        predictor = PlaintextPredictor(
            x=x,
            feature_names=feature_names,
            host_channel=host_channel
        )

        # 执行预测
        predictor.predict()


class PlaintextGuest:
    """明文训练的Guest实现"""

    def __init__(self,
                 x: np.ndarray,
                 feature_names: List[str],
                 host_channel: GrpcClient):
        self.x = x
        self.feature_names = feature_names
        self.host_channel = host_channel

        # 接收配置
        self.config = host_channel.recv('config')
        self.n_estimators = self.config['n_estimators']
        self.max_depth = self.config['max_depth']
        self.reg_lambda = self.config['reg_lambda']
        self.gamma = self.config['gamma']
        self.min_child_weight = self.config['min_child_weight']
        self.min_child_sample = self.config['min_child_sample']
        self.n_samples = self.config['n_samples']

    def train(self):
        """参与XGBoost训练"""
        for tree_idx in range(self.n_estimators):
            # 接收新树开始信号
            new_tree_msg = self.host_channel.recv('new_tree')
            logger.info(f"-------- Building tree {tree_idx + 1}/{self.n_estimators} --------")

            # 构建树
            self._build_tree()

            # 接收树完成信号
            self.host_channel.recv('tree_done')

        # 接收训练完成信号
        self.host_channel.recv('training_complete')

    def _build_tree(self):
        """参与单棵树的构建"""
        while True:
            # 接收节点处理请求
            node_msg = self.host_channel.recv('process_node')
            node_id = node_msg['node_id']
            sample_indices = node_msg['sample_indices']
            depth = node_msg['depth']

            # 接收节点结果（可能是叶子节点或需要分裂）
            # 首先检查是否直接标记为叶子
            if depth >= self.max_depth or len(sample_indices) < 2 * self.min_child_sample:
                # 等待Host发送叶子节点结果
                node_result = self.host_channel.recv('node_result')
                if node_result.get('is_leaf', True):
                    continue

            # 接收梯度和Hessian
            grad_hess = self.host_channel.recv('grad_hess')
            grad = np.array(grad_hess['grad'])
            hess = np.array(grad_hess['hess'])

            # 在本地特征上寻找最优分裂
            best_split = self._find_best_local_split(sample_indices, grad, hess)
            self.host_channel.send('best_split', best_split)

            # 接收分裂决策
            node_result = self.host_channel.recv('node_result')

            if node_result.get('is_leaf', True):
                continue

            # 如果是Guest方的特征被选中，计算样本划分
            if node_result['split_party'] == 'guest':
                split_feature = node_result['split_feature']
                split_value = node_result['split_value']

                if split_feature in self.feature_names:
                    feature_idx = self.feature_names.index(split_feature)
                    left_indices = [idx for idx in sample_indices
                                   if self.x[idx, feature_idx] <= split_value]
                    right_indices = [idx for idx in sample_indices
                                    if self.x[idx, feature_idx] > split_value]
                else:
                    # 不是这个Guest的特征
                    left_indices = []
                    right_indices = []

                self.host_channel.send('split_indices', {
                    'left_indices': left_indices,
                    'right_indices': right_indices
                })

    def _find_best_local_split(self,
                               sample_indices: List[int],
                               grad: np.ndarray,
                               hess: np.ndarray) -> Dict:
        """在Guest的本地特征中寻找最优分裂"""
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
                self.reg_lambda,
                self.gamma,
                self.min_child_weight,
                self.min_child_sample
            )

            if gain > best_split['gain']:
                best_split = {
                    'feature_idx': feature_idx,
                    'feature_name': self.feature_names[feature_idx],
                    'split_value': split_value,
                    'gain': gain
                }

        return best_split


class PlaintextPredictor:
    """明文预测的Guest实现"""

    def __init__(self,
                 x: np.ndarray,
                 feature_names: List[str],
                 host_channel: GrpcClient):
        self.x = x
        self.feature_names = feature_names
        self.host_channel = host_channel

    def predict(self):
        """参与预测"""
        # 接收预测配置
        config = self.host_channel.recv('predict_config')
        n_samples = config['n_samples']
        n_trees = config['n_trees']

        # 为每棵树参与预测
        for tree_idx in range(n_trees):
            # 接收开始预测树的信号
            self.host_channel.recv('predict_tree')

            # 持续响应分裂查询，直到收到树预测完成信号
            while True:
                # 尝试接收查询或完成信号
                query = self.host_channel.recv('query_split')
                if query is None:
                    # 可能是tree_predict_done信号
                    break

                sample_idx = query['sample_idx']
                split_feature = query['split_feature']
                split_value = query['split_value']

                if split_feature in self.feature_names:
                    feature_idx = self.feature_names.index(split_feature)
                    go_left = bool(self.x[sample_idx, feature_idx] <= split_value)
                    self.host_channel.send('go_left', {
                        'has_feature': True,
                        'go_left': go_left
                    })
                else:
                    self.host_channel.send('go_left', {
                        'has_feature': False,
                        'go_left': False
                    })

            # 接收树预测完成信号
            self.host_channel.recv('tree_predict_done')

        # 接收预测完成信号
        self.host_channel.recv('prediction_complete')
