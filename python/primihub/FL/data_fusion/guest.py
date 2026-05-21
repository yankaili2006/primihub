"""
Federated Learning Data Fusion Guest
联邦学习数据融合访客端

访客端负责提供特征数据参与融合
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.dataset import read_data

logger = logging.getLogger(__name__)


class DataFusionGuest(BaseModel):
    """数据融合访客端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 通用参数
        self.fusion_type = self.common_params.get("fusion_type", "vertical")
        self.fusion_method = self.common_params.get("fusion_method", "concat")
        self.secure = self.common_params.get("secure", False)

        # 角色参数
        self.data_info = self.role_params.get("data", {})
        self.selected_columns = self.role_params.get("selected_columns", [])
        self.id_column = self.role_params.get("id_column", None)

    def run(self):
        """执行数据融合"""
        logger.info("DataFusionGuest: 开始数据融合任务")

        # 获取host方
        parties = self.roles.keys()
        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        if host_party is None:
            raise ValueError("未找到host方")

        # 建立通信通道
        host_channel = GrpcClient(
            self.node_info["self_party"],
            host_party,
            self.node_info,
            self.task_info,
        )

        # 加载本地数据
        local_data = self._load_local_data()
        logger.info(f"DataFusionGuest: 加载本地数据 shape={local_data.shape}")

        # 根据融合类型执行不同操作
        if self.fusion_type == "vertical":
            self._vertical_fusion(local_data, host_channel)
        elif self.fusion_type == "horizontal":
            self._horizontal_fusion(local_data, host_channel)
        elif self.fusion_type == "statistics":
            self._statistics_fusion(local_data, host_channel)
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")

        logger.info("DataFusionGuest: 数据融合任务完成")

    def _load_local_data(self) -> np.ndarray:
        """加载本地数据"""
        if self.data_info:
            data = read_data(
                data_info=self.data_info,
                selected_column=self.selected_columns if self.selected_columns else None,
                droped_column=None,
            )
            return np.array(data)
        return np.array([])

    def _vertical_fusion(
        self, local_data: np.ndarray, host_channel: GrpcClient
    ):
        """纵向数据融合"""
        logger.info("DataFusionGuest: 参与纵向数据融合")

        # 接收样本ID进行对齐
        if self.id_column is not None:
            host_sample_ids = host_channel.recv("sample_ids")
            # 执行样本对齐
            local_data = self._align_samples(local_data, host_sample_ids)

        # 发送数据给host
        host_channel.send("guest_data", local_data.tolist())
        logger.info("DataFusionGuest: 已发送数据给host")

    def _horizontal_fusion(
        self, local_data: np.ndarray, host_channel: GrpcClient
    ):
        """横向数据融合"""
        logger.info("DataFusionGuest: 参与横向数据融合")

        # 发送数据给host
        host_channel.send("guest_data", local_data.tolist())
        logger.info("DataFusionGuest: 已发送数据给host")

    def _statistics_fusion(
        self, local_data: np.ndarray, host_channel: GrpcClient
    ):
        """统计融合"""
        logger.info("DataFusionGuest: 参与统计融合")

        # 计算本地统计量
        local_stats = {
            "mean": np.mean(local_data, axis=0).tolist(),
            "std": np.std(local_data, axis=0).tolist(),
            "min": np.min(local_data, axis=0).tolist(),
            "max": np.max(local_data, axis=0).tolist(),
            "count": int(local_data.shape[0]),
        }

        # 发送统计量给host
        host_channel.send("guest_stats", local_stats)
        host_channel.send("guest_sample_count", local_data.shape[0])
        logger.info("DataFusionGuest: 已发送统计量给host")

    def _align_samples(
        self, local_data: np.ndarray, host_sample_ids: List
    ) -> np.ndarray:
        """样本对齐"""
        if self.id_column is None or local_data.shape[0] == 0:
            return local_data

        # 假设第一列是ID列
        local_ids = local_data[:, 0].tolist()

        # 找到共同的样本
        common_ids = set(local_ids) & set(host_sample_ids)

        # 按照host的顺序对齐
        aligned_data = []
        for host_id in host_sample_ids:
            if host_id in common_ids:
                idx = local_ids.index(host_id)
                aligned_data.append(local_data[idx])

        return np.array(aligned_data) if aligned_data else local_data
