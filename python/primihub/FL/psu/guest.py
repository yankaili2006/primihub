"""
PSU Guest Implementation
隐私求并Guest端实现

Guest作为求并协议的参与方。
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from primihub.FL.utils.base import BaseModel
from primihub.FL.utils.file import save_pickle_file, save_json_file
from primihub.FL.utils.net_work import GrpcClient
from primihub.FL.utils.dataset import read_data

from .base import (
    HashBasedPSU,
    BloomFilterPSU,
    EncryptedPSU,
    SecurePSU,
)

logger = logging.getLogger(__name__)


class PSUGuest(BaseModel):
    """
    隐私求并Guest端

    执行联邦求并协议的Guest角色。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_params()

    def _parse_params(self):
        """解析参数"""
        # PSU协议类型
        self.protocol = self.common_params.get("protocol", "hash")
        # ID列名
        self.id_column = self.common_params.get("id_column", "id")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 输出模式
        self.output_mode = self.common_params.get("output_mode", "ids")

        # 布隆过滤器参数
        self.expected_elements = self.common_params.get("expected_elements", 100000)
        self.false_positive_rate = self.common_params.get("false_positive_rate", 0.01)

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

    def _create_psu(self, channel: Any):
        """创建PSU协议实例"""
        protocol_map = {
            "hash": HashBasedPSU,
            "bloom": BloomFilterPSU,
            "encrypted": EncryptedPSU,
            "secure": SecurePSU,
        }

        if self.protocol not in protocol_map:
            raise ValueError(f"Unknown PSU protocol: {self.protocol}")

        PSUClass = protocol_map[self.protocol]

        if self.protocol == "bloom":
            return PSUClass(
                role="guest",
                channel=channel,
                expected_elements=self.expected_elements,
                false_positive_rate=self.false_positive_rate,
            )
        else:
            return PSUClass(role="guest", channel=channel)

    def run(self):
        """执行隐私求并"""
        logger.info("PSUGuest: Starting Private Set Union")

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

        logger.info(f"PSUGuest: Data loaded, shape={data.shape}")

        # 提取ID列
        if self.id_column not in data.columns:
            logger.error(f"ID column '{self.id_column}' not found")
            return {"error": f"ID column '{self.id_column}' not found"}

        local_ids = set(data[self.id_column].dropna().unique())
        logger.info(f"PSUGuest: Local IDs count = {len(local_ids)}")

        # 创建PSU协议
        psu = self._create_psu(channel)

        # 执行求并
        try:
            union_result = psu.compute_union(local_ids)
        except Exception as e:
            logger.error(f"PSU computation failed: {e}")
            return {"error": str(e)}

        logger.info(f"PSUGuest: Union computed, identifiable elements = {len(union_result)}")

        # 等待Host完成信号
        try:
            complete = channel.recv("union_complete")
            logger.info("PSUGuest: Received completion signal from Host")
        except:
            pass

        # 准备结果
        result = {
            "protocol": self.protocol,
            "local_count": len(local_ids),
            "union_identifiable_count": len(union_result),
            "union_ids": list(union_result),
        }

        # 如果需要输出完整数据
        if self.output_mode == "data":
            # 筛选在并集中的数据
            result_data = data[data[self.id_column].isin(union_result)]
            result["data_shape"] = result_data.shape

            if self.output_path:
                if self.output_path.endswith('.csv'):
                    result_data.to_csv(self.output_path, index=False)
                else:
                    save_pickle_file(result_data, self.output_path)
                logger.info(f"PSUGuest: Result data saved to {self.output_path}")

        elif self.output_path:
            # 只保存ID
            if self.output_path.endswith('.json'):
                save_json_file({"union_ids": list(union_result)}, self.output_path)
            else:
                save_pickle_file(list(union_result), self.output_path)
            logger.info(f"PSUGuest: Union IDs saved to {self.output_path}")

        logger.info("PSUGuest: Completed")
        return result
