"""
Realtime Federated PSI Implementation
实时联邦求交实现

支持基于密钥交换DH算法的实时联邦求交：
- RealtimeDHPSI: 基于密钥交换DH算法的实时联邦求交

实时求交特点：
- 低延迟响应，适合在线场景
- 支持增量求交
- 会话密钥复用
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from .base import DHPSIProtocol

logger = logging.getLogger(__name__)


class RealtimeDHPSI(DHPSIProtocol):
    """
    基于密钥交换DH算法的实时联邦求交

    专为低延迟场景设计，支持：
    - 会话密钥复用
    - 增量求交
    - 流式处理

    协议流程：
    1. 首次求交时协商DH共享密钥（后续复用）
    2. 双方缓存加密后的数据索引
    3. 每次求交请求快速返回结果
    """

    def __init__(self, role: str, channel: Any = None,
                 prime_bits: int = 2048, session_timeout: int = 3600):
        """
        初始化实时DH求交

        Args:
            role: 角色
            channel: 通信通道
            prime_bits: DH素数位数
            session_timeout: 会话超时时间（秒）
        """
        super().__init__(role, channel, prime_bits)
        self.session_timeout = session_timeout
        self._session_start_time: Optional[float] = None
        self._is_session_active = False
        self._cached_local_hashes: Optional[Dict[str, Any]] = None
        self._cached_remote_hashes: Optional[Set[str]] = None

    def _is_session_valid(self) -> bool:
        """检查会话是否有效"""
        if not self._is_session_active or self._session_start_time is None:
            return False
        return (time.time() - self._session_start_time) < self.session_timeout

    def _establish_session(self, local_data: Set):
        """
        建立实时求交会话

        Args:
            local_data: 本地数据集
        """
        if self._is_session_valid() and self._cached_local_hashes is not None:
            logger.debug("RealtimeDHPSI: Reusing existing session")
            return

        logger.info(f"RealtimeDHPSI: {self.role} establishing new session")

        # 协商密钥
        self._negotiate_shared_key()

        # 缓存本地加密数据
        self._cached_local_hashes = self._encrypt_with_shared_key(local_data)

        # 交换加密数据索引
        local_hash_set = set(self._cached_local_hashes.keys())

        if self.role == "host":
            self.channel.send("realtime_psi_index_host", local_hash_set)
            self._cached_remote_hashes = self.channel.recv("realtime_psi_index_guest")
        else:
            self._cached_remote_hashes = self.channel.recv("realtime_psi_index_host")
            self.channel.send("realtime_psi_index_guest", local_hash_set)

        self._session_start_time = time.time()
        self._is_session_active = True

        logger.info(f"RealtimeDHPSI: {self.role} session established, "
                    f"local={len(local_hash_set)}, remote={len(self._cached_remote_hashes)}")

    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算交集（实时模式）

        Args:
            local_data: 本地数据集

        Returns:
            交集结果
        """
        logger.info(f"RealtimeDHPSI: {self.role} starting realtime PSI")

        self._local_set = self._to_set(local_data)

        # 建立或复用会话
        self._establish_session(self._local_set)

        # 快速计算交集
        local_hash_set = set(self._cached_local_hashes.keys())
        intersection_hashes = local_hash_set & self._cached_remote_hashes

        # 还原交集元素
        self._intersection_result = set()
        for h in intersection_hashes:
            if h in self._cached_local_hashes:
                self._intersection_result.add(self._cached_local_hashes[h])

        logger.info(f"RealtimeDHPSI: {self.role} intersection size = {len(self._intersection_result)}")

        return self._intersection_result

    def incremental_intersection(
        self,
        new_elements: Union[List, Set, np.ndarray]
    ) -> Set:
        """
        增量求交

        添加新元素到本地集合并计算与远程的交集

        Args:
            new_elements: 新增元素

        Returns:
            新元素与远程数据的交集
        """
        if not self._is_session_valid():
            logger.warning("RealtimeDHPSI: Session expired, need to re-establish")
            raise RuntimeError("Session expired")

        new_set = self._to_set(new_elements)

        # 加密新元素
        new_encrypted = self._encrypt_with_shared_key(new_set)

        # 更新本地缓存
        self._cached_local_hashes.update(new_encrypted)
        self._local_set.update(new_set)

        # 通知对方有新元素
        new_hashes = set(new_encrypted.keys())

        if self.role == "host":
            self.channel.send("realtime_psi_incremental_host", new_hashes)
            remote_new_hashes = self.channel.recv("realtime_psi_incremental_guest")
        else:
            remote_new_hashes = self.channel.recv("realtime_psi_incremental_host")
            self.channel.send("realtime_psi_incremental_guest", new_hashes)

        # 更新远程缓存
        self._cached_remote_hashes.update(remote_new_hashes)

        # 计算新元素的交集
        new_intersection_hashes = new_hashes & self._cached_remote_hashes

        # 还原结果
        result = set()
        for h in new_intersection_hashes:
            if h in new_encrypted:
                result.add(new_encrypted[h])

        # 更新总交集结果
        self._intersection_result.update(result)

        logger.info(f"RealtimeDHPSI: Incremental intersection found {len(result)} new matches")

        return result

    def check_membership(self, element: Any) -> bool:
        """
        检查单个元素是否在交集中

        Args:
            element: 要检查的元素

        Returns:
            是否在交集中
        """
        if not self._is_session_valid() or self._shared_key is None:
            raise RuntimeError("Session not established or expired")

        # 加密元素
        h = hmac.new(self._shared_key, str(element).encode(), hashlib.sha256).hexdigest()

        # 检查是否在远程数据中
        return h in self._cached_remote_hashes

    def stream_intersection(
        self,
        element_stream: Iterator[Any]
    ) -> Iterator[Any]:
        """
        流式求交

        Args:
            element_stream: 元素流

        Yields:
            在交集中的元素
        """
        if not self._is_session_valid():
            raise RuntimeError("Session not established or expired")

        for element in element_stream:
            if self.check_membership(element):
                yield element

    def refresh_session(self, local_data: Union[List, Set, np.ndarray]):
        """
        刷新会话

        强制重新建立会话，用于数据发生较大变化时

        Args:
            local_data: 新的本地数据集
        """
        logger.info(f"RealtimeDHPSI: {self.role} refreshing session")

        # 清除缓存
        self._cached_local_hashes = None
        self._cached_remote_hashes = None
        self._is_session_active = False

        # 通知对方刷新
        if self.role == "host":
            self.channel.send("realtime_psi_refresh", True)
            _ = self.channel.recv("realtime_psi_refresh_ack")
        else:
            _ = self.channel.recv("realtime_psi_refresh")
            self.channel.send("realtime_psi_refresh_ack", True)

        # 重新建立会话
        self._local_set = self._to_set(local_data)
        self._establish_session(self._local_set)

    def close_session(self):
        """关闭会话"""
        logger.info(f"RealtimeDHPSI: {self.role} closing session")

        self._is_session_active = False
        self._cached_local_hashes = None
        self._cached_remote_hashes = None
        self._shared_key = None

        if self.channel:
            try:
                if self.role == "host":
                    self.channel.send("realtime_psi_close", True)
                else:
                    self.channel.send("realtime_psi_close_ack", True)
            except Exception:
                pass  # 忽略关闭时的通信错误
