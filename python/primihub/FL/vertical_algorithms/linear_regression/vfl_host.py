"""
Vertical Federated Linear Regression - Host
Host方持有标签数据，负责计算误差并协调训练过程
"""
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import (
    save_json_file,
    save_pickle_file,
    load_pickle_file,
    save_csv_file
)
from primihub.FL.utils.dataset import read_data, DataLoader
from primihub.utils.logger_util import logger
from primihub.FL.crypto.ckks import CKKS
from primihub.FL.psi import sample_alignment
from primihub.FL.metrics import regression_metrics

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from .base import (
    LinearRegression_Host_Plaintext,
    LinearRegression_Host_CKKS
)


class VFLLinearRegressionHost(BaseModel):
    """纵向联邦线性回归 - Host方

    Host方持有标签数据，负责：
    1. 计算预测值（聚合各方的部分预测）
    2. 计算误差并分发给各Guest方
    3. 更新本地模型参数
    4. 保存模型和评估指标
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

        method = self.common_params['method']
        if method == 'CKKS':
            coordinator_channel = GrpcClient(
                local_party=self.role_params['self_name'],
                remote_party=self.roles['coordinator'],
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
        x = check_array(x, dtype='numeric')

        # host init
        batch_size = min(x.shape[0], self.common_params['batch_size'])
        learning_rate = self.common_params['learning_rate']
        alpha = self.common_params['alpha']

        if method == 'Plaintext':
            host = PlaintextHost(x, learning_rate, alpha, guest_channel)
        elif method == 'CKKS':
            coordinator_channel.send('batch_size', batch_size)
            host = CKKSHost(
                x, learning_rate, alpha,
                guest_channel, coordinator_channel
            )
        else:
            error_msg = f"Unsupported method: {method}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # data preprocessing - StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # host training
        train_dataloader = DataLoader(
            dataset=x,
            label=y,
            batch_size=batch_size,
            shuffle=True,
            seed=self.common_params['shuffle_seed']
        )

        logger.info("-------- start training --------")
        epoch = self.common_params['epoch']
        for i in range(epoch):
            logger.info(f"-------- epoch {i+1} / {epoch} --------")
            for batch_x, batch_y in train_dataloader:
                host.train(batch_x, batch_y)

            # print metrics
            if self.common_params['print_metrics']:
                host.compute_metrics(x, y)
        logger.info("-------- finish training --------")

        # receive plaintext model
        if method == 'CKKS':
            host.update_plaintext_model()

        # compute final metrics
        train_metrics = host.compute_final_metrics(x, y)
        save_json_file(train_metrics, self.role_params['metric_path'])

        # save model for prediction
        model_file = {
            "selected_column": selected_column,
            "id": id_col,
            "label": label,
            "transformer": scaler,
            "model": host.model
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
        x = check_array(x, dtype='numeric')

        # data preprocessing
        transformer = model_file['transformer']
        x = transformer.transform(x)

        # test data prediction
        model = model_file['model']
        guest_z = guest_channel.recv_all('guest_z')
        z = model.compute_z(x, guest_z)

        result = pd.DataFrame({'pred_y': z})

        data_result = pd.concat([origin_data, result], axis=1)
        save_csv_file(data_result, self.role_params['predict_path'])


class PlaintextHost:
    """明文训练的Host实现"""

    def __init__(self, x, learning_rate, alpha, guest_channel):
        self.model = LinearRegression_Host_Plaintext(x, learning_rate, alpha)
        self.guest_channel = guest_channel

    def compute_z(self, x):
        guest_z = self.guest_channel.recv_all('guest_z')
        return self.model.compute_z(x, guest_z)

    def train(self, x, y):
        z = self.compute_z(x)

        error = self.model.compute_error(y, z)
        self.guest_channel.send_all('error', error)

        self.model.fit(x, error)

    def compute_metrics(self, x, y):
        y_pred = self.compute_z(x)
        regression_metrics(
            y,
            y_pred,
            metircs_name=["mae", "mse"],
        )

    def compute_final_metrics(self, x, y):
        y_pred = self.compute_z(x)
        metrics = regression_metrics(
            y,
            y_pred,
            prefix="train_",
            metircs_name=[
                "ev", "maxe", "mae", "mse",
                "rmse", "medae", "r2"
            ],
        )
        return metrics


class CKKSHost(PlaintextHost, CKKS):
    """CKKS同态加密训练的Host实现"""

    def __init__(self, x, learning_rate, alpha,
                 guest_channel, coordinator_channel):
        self.t = 0
        self.model = LinearRegression_Host_CKKS(x, learning_rate, alpha)
        self.guest_channel = guest_channel
        self.recv_public_context(coordinator_channel)
        coordinator_channel.send('num_examples', x.shape[0])
        CKKS.__init__(self, self.context)
        multiply_per_iter = 2
        self.max_iter = self.multiply_depth // multiply_per_iter
        self.encrypt_model()

    def recv_public_context(self, coordinator_channel):
        self.coordinator_channel = coordinator_channel
        self.context = coordinator_channel.recv('public_context')

    def encrypt_model(self):
        self.model.weight = self.encrypt_vector(self.model.weight)
        self.model.bias = self.encrypt_vector(self.model.bias)

    def update_ciphertext_model(self):
        self.coordinator_channel.send(
            'host_weight', self.model.weight.serialize()
        )
        self.coordinator_channel.send(
            'host_bias', self.model.bias.serialize()
        )

        self.model.weight = self.load_vector(
            self.coordinator_channel.recv('host_weight')
        )
        self.model.bias = self.load_vector(
            self.coordinator_channel.recv('host_bias')
        )

    def update_plaintext_model(self):
        self.coordinator_channel.send(
            'host_weight', self.model.weight.serialize()
        )
        self.coordinator_channel.send(
            'host_bias', self.model.bias.serialize()
        )
        self.model.weight = self.coordinator_channel.recv('host_weight')
        self.model.bias = self.coordinator_channel.recv('host_bias')

    def compute_enc_z(self, x):
        guest_z = self.guest_channel.recv_all('guest_z')
        guest_z = [self.load_vector(z) for z in guest_z]
        return self.model.compute_enc_z(x, guest_z)

    def train(self, x, y):
        logger.info(f'iteration {self.t} / {self.max_iter}')
        if self.t >= self.max_iter:
            self.t = 0
            logger.warning('decrypt model')
            self.update_ciphertext_model()

        self.t += 1

        z = self.compute_enc_z(x)

        error = self.model.compute_error(y, z)
        self.guest_channel.send_all('error', error.serialize())

        self.model.fit(x, error)

    def compute_metrics(self, x, y):
        logger.info(f'iteration {self.t} / {self.max_iter}')
        if self.t >= self.max_iter:
            self.t = 0
            logger.warning('decrypt model')
            self.update_ciphertext_model()

        z = self.compute_enc_z(x)

        mse = ((z - y) ** 2).sum()
        self.coordinator_channel.send('mse', mse.serialize())
        logger.info('View metrics at coordinator while using CKKS')
