"""
Vertical Federated XGBoost - Coordinator
Coordinator负责管理安全增强版本的密钥（可选）
对于明文版本，Coordinator主要用于协调和监控训练过程
"""
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.base import BaseModel
from primihub.utils.logger_util import logger
from primihub.FL.crypto.ckks import CKKS

import math
import numpy as np
import tenseal as ts
from typing import Dict, List, Optional


class VFLXGBoostCoordinator(BaseModel):
    """纵向联邦XGBoost - Coordinator

    Coordinator负责：
    1. 管理CKKS同态加密密钥（安全增强版本）
    2. 协调安全梯度聚合
    3. 监控训练过程

    注意：对于明文版本，Coordinator是可选的
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        process = self.common_params['process']
        logger.info(f"process: {process}")
        if process == 'train':
            self.train()
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
        guest_channel = MultiGrpcClients(
            local_party=self.role_params['self_name'],
            remote_parties=self.roles['guest'],
            node_info=self.node_info,
            task_info=self.task_info
        )

        # coordinator init
        method = self.common_params.get('method', 'Plaintext')
        if method == 'CKKS':
            coordinator = CKKSCoordinator(host_channel, guest_channel)
        else:
            coordinator = PlaintextCoordinator(host_channel, guest_channel)

        # coordinator training
        logger.info("-------- start training --------")
        coordinator.train()
        logger.info("-------- finish training --------")


class PlaintextCoordinator:
    """明文版本的Coordinator实现

    对于明文XGBoost，Coordinator主要用于监控训练过程
    """

    def __init__(self, host_channel: GrpcClient, guest_channel: MultiGrpcClients):
        self.host_channel = host_channel
        self.guest_channel = guest_channel

    def train(self):
        """监控训练过程"""
        logger.info("Plaintext XGBoost - Coordinator is monitoring training...")

        # 接收训练配置
        config = self.host_channel.recv('coordinator_config')
        if config is None:
            logger.info("No coordinator config received, training in plaintext mode")
            return

        n_estimators = config.get('n_estimators', 100)

        for tree_idx in range(n_estimators):
            # 接收树训练状态
            status = self.host_channel.recv('tree_status')
            if status:
                logger.info(f"Tree {tree_idx + 1}: {status}")

        logger.info("Training completed")


class CKKSCoordinator(CKKS):
    """CKKS同态加密的Coordinator实现

    用于安全增强版本的XGBoost，保护梯度信息
    """

    def __init__(self, host_channel: GrpcClient, guest_channel: MultiGrpcClients):
        self.host_channel = host_channel
        self.guest_channel = guest_channel

        # set CKKS params
        poly_mod_degree = 8192
        multiply_per_iter = 2
        self.max_iter = 1
        multiply_depth = multiply_per_iter * self.max_iter
        fe_bits_scale = 60
        bits_scale = 49
        coeff_mod_bit_sizes = (
            [fe_bits_scale] +
            [bits_scale] * multiply_depth +
            [fe_bits_scale]
        )

        # create TenSEALContext
        logger.info('create CKKS TenSEAL context')
        secret_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        secret_context.global_scale = pow(2, bits_scale)
        secret_context.generate_galois_keys()

        context = secret_context.copy()
        context.make_context_public()

        super().__init__(context)
        self.secret_context = secret_context

        self.send_public_context()

    def send_public_context(self):
        """发送公钥给各参与方"""
        serialize_context = self.context.serialize()
        self.host_channel.send("public_context", serialize_context)
        self.guest_channel.send_all("public_context", serialize_context)

    def train(self):
        """安全增强版本的训练协调"""
        logger.info("CKKS XGBoost - Coordinator managing secure training...")

        # 接收训练配置
        config = self.host_channel.recv('coordinator_config')
        n_estimators = config.get('n_estimators', 100)

        for tree_idx in range(n_estimators):
            logger.info(f"-------- Coordinating tree {tree_idx + 1}/{n_estimators} --------")
            self._coordinate_tree_building()

        # 最终解密和同步
        self._finalize_training()

    def _coordinate_tree_building(self):
        """协调单棵树的构建"""
        while True:
            # 接收是否需要处理加密梯度
            msg = self.host_channel.recv('encrypt_request')
            if msg is None or msg.get('tree_done', False):
                break

            if msg.get('decrypt_gradients', False):
                # 解密梯度用于分裂查找
                enc_gradients = self.host_channel.recv('encrypted_gradients')
                if enc_gradients:
                    dec_gradients = self._decrypt_gradients(enc_gradients)
                    self.host_channel.send('decrypted_gradients', dec_gradients)

    def _decrypt_gradients(self, enc_gradients: Dict) -> Dict:
        """解密梯度信息"""
        secret_key = self.secret_context.secret_key()

        enc_g = self.load_vector(enc_gradients['grad'])
        enc_h = self.load_vector(enc_gradients['hess'])

        grad = np.array(self.decrypt(enc_g, secret_key))
        hess = np.array(self.decrypt(enc_h, secret_key))

        return {'grad': grad, 'hess': hess}

    def _finalize_training(self):
        """完成训练的最终处理"""
        logger.info("Finalizing secure training...")
        self.host_channel.send('training_finalized', True)
        self.guest_channel.send_all('training_finalized', True)
