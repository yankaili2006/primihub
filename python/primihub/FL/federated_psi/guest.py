"""
Federated PSI Guest Implementation
联邦求交Guest端实现

Guest作为求交参与方，响应Host的求交请求。
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.dataset import read_data

from .batch_psi import BatchDHPSI, BatchOTPSI, BatchHEPSI
from .realtime_psi import RealtimeDHPSI

logger = logging.getLogger(__name__)


# 协议和模式到类的映射
PSI_PROTOCOL_MAP = {
    ("dh", "batch"): BatchDHPSI,
    ("ot", "batch"): BatchOTPSI,
    ("he", "batch"): BatchHEPSI,
    ("dh", "realtime"): RealtimeDHPSI,
}


class FederatedPSIGuest(BaseModel):
    """
    联邦求交Guest端

    执行联邦求交协议的Guest角色。

    支持的协议：
    - dh: 基于密钥交换DH算法
    - ot: 基于不经意传输OT算法
    - he: 基于全同态加密HE算法

    支持的模式：
    - batch: 批量求交
    - realtime: 实时求交（仅DH协议支持）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # PSI协议
        self.protocol = self.common_params.get("protocol", "dh").lower()
        # PSI模式
        self.mode = self.common_params.get("mode", "batch").lower()
        # ID列名
        self.id_column = self.common_params.get("id_column", "id")
        # 输出路径（Guest可选）
        self.output_path = self.common_params.get("output_path", "")
        # 输出模式
        self.output_mode = self.common_params.get("output_mode", "ids")
        # 输出格式
        self.output_format = self.common_params.get("output_format", "pickle")

        # 协议特定参数
        self.batch_size = self.common_params.get("batch_size", 10000)
        self.security_parameter = self.common_params.get("security_parameter", 128)
        self.key_size = self.common_params.get("key_size", 2048)

        # 数据信息
        self.data_info = self.role_params.get("data", {})

    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        if not self.data_info:
            logger.warning("No data info provided")
            return pd.DataFrame()

        data = read_data(data_info=self.data_info)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        return data

    def _create_psi_protocol(self, channel: Any):
        """创建PSI协议实例"""
        protocol_key = (self.protocol, self.mode)

        if protocol_key not in PSI_PROTOCOL_MAP:
            if self.mode == "realtime" and self.protocol in ("ot", "he"):
                logger.warning(f"Protocol {self.protocol} does not support realtime mode, "
                               f"falling back to batch mode")
                protocol_key = (self.protocol, "batch")
            else:
                raise ValueError(
                    f"Unknown protocol/mode combination: {self.protocol}/{self.mode}. "
                    f"Supported: {list(PSI_PROTOCOL_MAP.keys())}"
                )

        PSIClass = PSI_PROTOCOL_MAP[protocol_key]

        if self.protocol == "dh":
            return PSIClass(role="guest", channel=channel)
        elif self.protocol == "ot":
            return PSIClass(
                role="guest",
                channel=channel,
                security_parameter=self.security_parameter
            )
        elif self.protocol == "he":
            return PSIClass(
                role="guest",
                channel=channel,
                key_size=self.key_size
            )
        else:
            return PSIClass(role="guest", channel=channel)

    def _save_results(self, intersection_ids: Set, data: pd.DataFrame):
        """保存求交结果"""
        if not self.output_path:
            return

        if self.output_mode == "data":
            result_data = data[data[self.id_column].isin(intersection_ids)]

            if self.output_format == "csv":
                result_data.to_csv(self.output_path, index=False)
            else:
                save_pickle_file(result_data, self.output_path)
        else:
            result = {
                "protocol": self.protocol,
                "mode": self.mode,
                "intersection_count": len(intersection_ids),
                "intersection_ids": list(intersection_ids),
            }

            if self.output_format == "json":
                save_json_file(result, self.output_path)
            else:
                save_pickle_file(result, self.output_path)

        logger.info(f"FederatedPSIGuest: Results saved to {self.output_path}")

    def run(self):
        """执行联邦求交（Guest端）"""
        logger.info(f"FederatedPSIGuest: Starting federated PSI "
                    f"(protocol={self.protocol}, mode={self.mode})")

        # 找到Host
        host_party = None
        for party, role in self.roles.items():
            if role == "host":
                host_party = party
                break

        if not host_party:
            logger.error("No host party found")
            return {"error": "No host party found"}

        # 建立通信通道
        channel = GrpcClient(
            self.node_info["self_party"],
            host_party,
            self.node_info,
            self.task_info,
        )

        # 加载数据
        data = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"FederatedPSIGuest: Data loaded, shape={data.shape}")

        # 提取ID列
        if self.id_column not in data.columns:
            logger.error(f"ID column '{self.id_column}' not found")
            return {"error": f"ID column '{self.id_column}' not found"}

        local_ids = set(data[self.id_column].dropna().unique())
        logger.info(f"FederatedPSIGuest: Local IDs count = {len(local_ids)}")

        # 创建PSI协议
        try:
            psi_protocol = self._create_psi_protocol(channel)
        except ValueError as e:
            logger.error(f"Failed to create PSI protocol: {e}")
            return {"error": str(e)}

        # 执行求交
        try:
            intersection_result = psi_protocol.compute_intersection(local_ids)
        except Exception as e:
            logger.error(f"PSI computation failed: {e}")
            return {"error": str(e)}

        logger.info(f"FederatedPSIGuest: Intersection computed, "
                    f"size = {len(intersection_result)}")

        # 保存结果
        self._save_results(intersection_result, data)

        return {
            "protocol": self.protocol,
            "mode": self.mode,
            "local_count": len(local_ids),
            "intersection_count": len(intersection_result),
            "output_path": self.output_path if self.output_path else None,
        }
