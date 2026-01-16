"""
Federated Query Host Implementation
联邦查询Host端实现

Host作为查询发起方，支持批量和实时两种查询模式。
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.net_work import GrpcClient, MultiGrpcClients
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


class FederatedQueryHost(BaseModel):
    """
    联邦查询Host端

    执行联邦查询协议的Host角色（查询发起方）。

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
        # 查询协议: dh, ot, he
        self.protocol = self.common_params.get("protocol", "dh").lower()
        # 查询模式: batch, realtime
        self.mode = self.common_params.get("mode", "batch").lower()
        # 查询键列名
        self.key_column = self.common_params.get("key_column", "id")
        # 值列名（可选，为空则返回整行）
        self.value_columns = self.common_params.get("value_columns", None)
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 输出格式: json, csv, pickle
        self.output_format = self.common_params.get("output_format", "pickle")

        # 协议特定参数
        self.batch_size = self.common_params.get("batch_size", 10000)
        self.security_parameter = self.common_params.get("security_parameter", 128)
        self.key_size = self.common_params.get("key_size", 2048)

        # 数据信息
        self.data_info = self.role_params.get("data", {})

    def _load_data(self) -> pd.DataFrame:
        """加载查询数据"""
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
            return QueryClass(role="host", channel=channel)
        elif self.protocol == "ot":
            return QueryClass(
                role="host",
                channel=channel,
                security_parameter=self.security_parameter
            )
        elif self.protocol == "he":
            return QueryClass(
                role="host",
                channel=channel,
                key_size=self.key_size
            )
        else:
            return QueryClass(role="host", channel=channel)

    def _prepare_query_data(self, data: pd.DataFrame) -> tuple:
        """
        准备查询数据

        Returns:
            (查询键集合, 键到数据的映射)
        """
        if self.key_column not in data.columns:
            raise ValueError(f"Key column '{self.key_column}' not found in data")

        # 提取查询键
        query_keys = set(data[self.key_column].dropna().unique())

        # 构建键到数据的映射
        if self.value_columns:
            # 只保留指定的值列
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

        return query_keys, local_data

    def _save_results(self, results: Dict[Any, Any], matched_data: pd.DataFrame):
        """保存查询结果"""
        if not self.output_path:
            return

        if self.output_format == "json":
            # JSON格式
            output = {
                "protocol": self.protocol,
                "mode": self.mode,
                "matched_count": len(results),
                "results": [
                    {"key": k, "value": v}
                    for k, v in results.items()
                ]
            }
            save_json_file(output, self.output_path)

        elif self.output_format == "csv":
            # CSV格式
            if not matched_data.empty:
                matched_data.to_csv(self.output_path, index=False)
            else:
                pd.DataFrame().to_csv(self.output_path, index=False)

        else:
            # Pickle格式
            save_pickle_file({
                "results": results,
                "data": matched_data
            }, self.output_path)

        logger.info(f"FederatedQueryHost: Results saved to {self.output_path}")

    def run(self):
        """执行联邦查询"""
        logger.info(f"FederatedQueryHost: Starting federated query "
                    f"(protocol={self.protocol}, mode={self.mode})")

        # 获取Guest列表
        guest_parties = []
        for party, role in self.roles.items():
            if role == "guest":
                guest_parties.append(party)

        if not guest_parties:
            logger.error("No guest parties found")
            return {"error": "No guest parties found"}

        # 建立通信通道
        if len(guest_parties) > 1:
            channel = MultiGrpcClients(
                self.node_info["self_party"],
                guest_parties,
                self.node_info,
                self.task_info,
            )
        else:
            channel = GrpcClient(
                self.node_info["self_party"],
                guest_parties[0],
                self.node_info,
                self.task_info,
            )

        # 加载数据
        data = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"FederatedQueryHost: Data loaded, shape={data.shape}")

        # 准备查询数据
        try:
            query_keys, local_data = self._prepare_query_data(data)
        except ValueError as e:
            logger.error(f"Failed to prepare query data: {e}")
            return {"error": str(e)}

        logger.info(f"FederatedQueryHost: Prepared {len(query_keys)} query keys")

        # 创建查询协议
        try:
            query_protocol = self._create_query_protocol(channel)
        except ValueError as e:
            logger.error(f"Failed to create query protocol: {e}")
            return {"error": str(e)}

        # 执行查询
        try:
            results = query_protocol.execute_query(query_keys, local_data)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": str(e)}

        logger.info(f"FederatedQueryHost: Query completed, "
                    f"matched {len(results)} records")

        # 构建匹配的数据DataFrame
        matched_keys = set(results.keys())
        matched_data = data[data[self.key_column].isin(matched_keys)].copy()

        # 保存结果
        self._save_results(results, matched_data)

        # 返回结果摘要
        return {
            "protocol": self.protocol,
            "mode": self.mode,
            "input_count": len(query_keys),
            "matched_count": len(results),
            "output_path": self.output_path if self.output_path else None,
        }
