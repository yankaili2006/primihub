"""
Federated Query Guest Implementation
联邦查询Guest端实现

Guest作为数据提供方，响应Host的查询请求。
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.dataset import read_data

from .batch_query import BatchDHQuery, BatchOTQuery, BatchHEQuery
from .realtime_query import RealtimeDHQuery, RealtimeOTQuery, RealtimeHEQuery

logger = logging.getLogger(__name__)


# 协议和模式到类的映射
QUERY_PROTOCOL_MAP = {
    ("dh", "batch"): BatchDHQuery,
    ("ot", "batch"): BatchOTQuery,
    ("he", "batch"): BatchHEQuery,
    ("dh", "realtime"): RealtimeDHQuery,
    ("ot", "realtime"): RealtimeOTQuery,
    ("he", "realtime"): RealtimeHEQuery,
}


class FederatedQueryGuest(BaseModel):
    """
    联邦查询Guest端

    执行联邦查询协议的Guest角色（数据提供方）。

    支持的协议：
    - dh: 基于密钥交换DH算法
    - ot: 基于不经意传输OT算法
    - he: 基于全同态加密HE算法

    支持的模式：
    - batch: 批量查询
    - realtime: 实时查询
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # 查询协议
        self.protocol = self.common_params.get("protocol", "dh").lower()
        # 查询模式
        self.mode = self.common_params.get("mode", "batch").lower()
        # 数据键列名
        self.key_column = self.common_params.get("key_column", "id")
        # 值列名
        self.value_columns = self.common_params.get("value_columns", None)
        # 输出路径（Guest通常不需要输出，但可选）
        self.output_path = self.common_params.get("output_path", "")

        # 协议特定参数
        self.batch_size = self.common_params.get("batch_size", 10000)
        self.security_parameter = self.common_params.get("security_parameter", 128)
        self.key_size = self.common_params.get("key_size", 2048)

        # 数据信息
        self.data_info = self.role_params.get("data", {})

    def _load_data(self) -> pd.DataFrame:
        """加载本地数据"""
        if not self.data_info:
            logger.warning("No data info provided")
            return pd.DataFrame()

        data = read_data(data_info=self.data_info)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        return data

    def _create_query_protocol(self, channel: Any):
        """创建查询协议实例"""
        protocol_key = (self.protocol, self.mode)

        if protocol_key not in QUERY_PROTOCOL_MAP:
            raise ValueError(
                f"Unknown protocol/mode combination: {self.protocol}/{self.mode}. "
                f"Supported: {list(QUERY_PROTOCOL_MAP.keys())}"
            )

        QueryClass = QUERY_PROTOCOL_MAP[protocol_key]

        # 根据协议类型传递参数
        if self.protocol == "dh":
            return QueryClass(role="guest", channel=channel)
        elif self.protocol == "ot":
            return QueryClass(
                role="guest",
                channel=channel,
                security_parameter=self.security_parameter
            )
        elif self.protocol == "he":
            return QueryClass(
                role="guest",
                channel=channel,
                key_size=self.key_size
            )
        else:
            return QueryClass(role="guest", channel=channel)

    def _prepare_data(self, data: pd.DataFrame) -> tuple:
        """
        准备本地数据

        Returns:
            (数据键集合, 键到数据的映射)
        """
        if self.key_column not in data.columns:
            raise ValueError(f"Key column '{self.key_column}' not found in data")

        # 提取数据键
        data_keys = set(data[self.key_column].dropna().unique())

        # 构建键到数据的映射
        if self.value_columns:
            cols = [self.key_column] + [c for c in self.value_columns if c in data.columns]
            data_subset = data[cols]
        else:
            data_subset = data

        # 创建键到行数据的映射
        local_data = {}
        for _, row in data_subset.iterrows():
            key = row[self.key_column]
            if pd.notna(key):
                local_data[key] = row.to_dict()

        return data_keys, local_data

    def run(self):
        """执行联邦查询（Guest端）"""
        logger.info(f"FederatedQueryGuest: Starting federated query "
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

        logger.info(f"FederatedQueryGuest: Data loaded, shape={data.shape}")

        # 准备本地数据
        try:
            data_keys, local_data = self._prepare_data(data)
        except ValueError as e:
            logger.error(f"Failed to prepare data: {e}")
            return {"error": str(e)}

        logger.info(f"FederatedQueryGuest: Prepared {len(data_keys)} data records")

        # 创建查询协议
        try:
            query_protocol = self._create_query_protocol(channel)
        except ValueError as e:
            logger.error(f"Failed to create query protocol: {e}")
            return {"error": str(e)}

        # 执行查询协议
        try:
            results = query_protocol.execute_query(data_keys, local_data)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": str(e)}

        logger.info("FederatedQueryGuest: Query protocol completed")

        # Guest通常不保存结果（隐私保护）
        # 但如果配置了输出路径，可以保存参与统计信息
        if self.output_path:
            stats = {
                "protocol": self.protocol,
                "mode": self.mode,
                "local_data_count": len(data_keys),
            }
            save_json_file(stats, self.output_path)
            logger.info(f"FederatedQueryGuest: Stats saved to {self.output_path}")

        return {
            "protocol": self.protocol,
            "mode": self.mode,
            "local_data_count": len(data_keys),
        }
