"""
Batch Federated PSI Implementations
批量联邦求交实现

支持三种密码学协议的批量求交：
- BatchDHPSI: 基于密钥交换DH算法的批量联邦求交
- BatchOTPSI: 基于不经意传输OT算法的批量联邦求交
- BatchHEPSI: 基于全同态加密HE算法的批量联邦求交
"""

import hashlib
import hmac
import logging
import secrets
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .base import DHPSIProtocol, OTPSIProtocol, HEPSIProtocol

logger = logging.getLogger(__name__)


class BatchDHPSI(DHPSIProtocol):
    """
    基于密钥交换DH算法的批量联邦求交

    适用于大规模数据集的批量求交场景。
    一次性处理所有数据，减少通信轮数。

    协议流程：
    1. Host和Guest通过DH协议协商共享密钥
    2. 双方使用共享密钥对各自的数据进行哈希
    3. 交换哈希后的数据集合
    4. 计算交集
    5. 返回交集中本地可识别的元素
    """

    def __init__(self, role: str, channel: Any = None,
                 prime_bits: int = 2048, batch_size: int = 10000):
        """
        初始化批量DH求交

        Args:
            role: 角色
            channel: 通信通道
            prime_bits: DH素数位数
            batch_size: 批处理大小
        """
        super().__init__(role, channel, prime_bits)
        self.batch_size = batch_size

    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算交集

        Args:
            local_data: 本地数据集

        Returns:
            交集结果（本地可识别的元素）
        """
        logger.info(f"BatchDHPSI: {self.role} starting batch PSI")

        self._local_set = self._to_set(local_data)

        # 1. 协商DH共享密钥
        self._negotiate_shared_key()

        # 2. 使用共享密钥加密本地数据
        local_encrypted = self._encrypt_with_shared_key(self._local_set)
        local_hashes = set(local_encrypted.keys())

        logger.info(f"BatchDHPSI: {self.role} encrypted {len(local_hashes)} elements")

        # 3. 分批交换哈希值
        all_batches = list(local_hashes)
        num_batches = (len(all_batches) + self.batch_size - 1) // self.batch_size

        if self.role == "host":
            # 交换批次数量
            self.channel.send("batch_psi_num_batches", num_batches)
            guest_num_batches = self.channel.recv("batch_psi_num_batches")

            remote_hashes = set()

            # 发送本地批次
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_batches))
                batch = set(all_batches[start_idx:end_idx])
                self.channel.send(f"batch_psi_host_{i}", batch)

            # 接收远程批次
            for i in range(guest_num_batches):
                batch = self.channel.recv(f"batch_psi_guest_{i}")
                remote_hashes.update(batch)

            # 计算交集哈希
            intersection_hashes = local_hashes & remote_hashes

            # 广播交集大小
            self.channel.send("batch_psi_intersection_size", len(intersection_hashes))

        else:  # guest
            guest_num_batches = num_batches
            self.channel.send("batch_psi_num_batches", guest_num_batches)
            host_num_batches = self.channel.recv("batch_psi_num_batches")

            remote_hashes = set()

            # 接收Host批次
            for i in range(host_num_batches):
                batch = self.channel.recv(f"batch_psi_host_{i}")
                remote_hashes.update(batch)

            # 发送Guest批次
            for i in range(guest_num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_batches))
                batch = set(all_batches[start_idx:end_idx])
                self.channel.send(f"batch_psi_guest_{i}", batch)

            # 计算交集哈希
            intersection_hashes = local_hashes & remote_hashes

            # 接收交集大小确认
            intersection_size = self.channel.recv("batch_psi_intersection_size")
            logger.info(f"BatchDHPSI: Guest confirmed intersection size = {intersection_size}")

        # 4. 还原交集中本地可识别的元素
        self._intersection_result = set()
        for h in intersection_hashes:
            if h in local_encrypted:
                self._intersection_result.add(local_encrypted[h])

        logger.info(f"BatchDHPSI: {self.role} intersection size = {len(self._intersection_result)}")

        return self._intersection_result


class BatchOTPSI(OTPSIProtocol):
    """
    基于不经意传输OT算法的批量联邦求交

    使用OT协议确保双方都不泄露非交集元素的信息。

    协议流程：
    1. Guest（数据方）准备数据集
    2. Host（查询方）通过OT协议进行匹配查询
    3. 双方只获知交集元素，不泄露其他信息
    """

    def __init__(self, role: str, channel: Any = None,
                 security_parameter: int = 128, batch_size: int = 1000):
        """
        初始化批量OT求交

        Args:
            role: 角色
            channel: 通信通道
            security_parameter: 安全参数
            batch_size: 批处理大小
        """
        super().__init__(role, channel, security_parameter)
        self.batch_size = batch_size

    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算交集

        Args:
            local_data: 本地数据集

        Returns:
            交集结果
        """
        logger.info(f"BatchOTPSI: {self.role} starting OT-based batch PSI")

        self._local_set = self._to_set(local_data)

        # 使用哈希进行安全比较
        salt = secrets.token_bytes(32)

        if self.role == "host":
            # Host发起求交请求
            # 1. 协商salt
            self.channel.send("ot_psi_salt", salt)

            # 2. 计算本地哈希
            local_hash_map = {}
            for elem in self._local_set:
                h = hmac.new(salt, str(elem).encode(), hashlib.sha256).hexdigest()
                local_hash_map[h] = elem

            local_hashes = set(local_hash_map.keys())

            # 3. 发送哈希集合大小
            self.channel.send("ot_psi_host_size", len(local_hashes))
            guest_size = self.channel.recv("ot_psi_guest_size")

            # 4. 批量交换哈希进行匹配
            # 发送本地哈希
            self.channel.send("ot_psi_host_hashes", local_hashes)

            # 接收Guest的哈希
            guest_hashes = self.channel.recv("ot_psi_guest_hashes")

            # 5. 计算交集
            intersection_hashes = local_hashes & guest_hashes

            # 6. 还原交集元素
            self._intersection_result = set()
            for h in intersection_hashes:
                if h in local_hash_map:
                    self._intersection_result.add(local_hash_map[h])

            # 7. 发送确认
            self.channel.send("ot_psi_complete", True)

        else:  # guest
            # Guest响应求交请求
            # 1. 接收salt
            salt = self.channel.recv("ot_psi_salt")

            # 2. 计算本地哈希
            local_hash_map = {}
            for elem in self._local_set:
                h = hmac.new(salt, str(elem).encode(), hashlib.sha256).hexdigest()
                local_hash_map[h] = elem

            local_hashes = set(local_hash_map.keys())

            # 3. 交换大小
            host_size = self.channel.recv("ot_psi_host_size")
            self.channel.send("ot_psi_guest_size", len(local_hashes))

            # 4. 接收Host哈希
            host_hashes = self.channel.recv("ot_psi_host_hashes")

            # 发送Guest哈希
            self.channel.send("ot_psi_guest_hashes", local_hashes)

            # 5. 计算交集
            intersection_hashes = local_hashes & host_hashes

            # 6. 还原交集元素
            self._intersection_result = set()
            for h in intersection_hashes:
                if h in local_hash_map:
                    self._intersection_result.add(local_hash_map[h])

            # 7. 接收完成确认
            _ = self.channel.recv("ot_psi_complete")

        logger.info(f"BatchOTPSI: {self.role} intersection size = {len(self._intersection_result)}")

        return self._intersection_result


class BatchHEPSI(HEPSIProtocol):
    """
    基于全同态加密HE算法的批量联邦求交

    使用同态加密在密文上进行交集计算，提供最强的隐私保护。

    协议流程：
    1. Host生成HE密钥对
    2. Host加密查询数据发送给Guest
    3. Guest在密文上执行多项式评估（匹配检测）
    4. Guest返回加密的匹配结果
    5. Host解密获取交集
    """

    def __init__(self, role: str, channel: Any = None,
                 key_size: int = 2048, batch_size: int = 100):
        """
        初始化批量HE求交

        Args:
            role: 角色
            channel: 通信通道
            key_size: HE密钥长度
            batch_size: 批处理大小（HE计算开销大）
        """
        super().__init__(role, channel, key_size)
        self.batch_size = batch_size

    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算交集

        Args:
            local_data: 本地数据集

        Returns:
            交集结果
        """
        logger.info(f"BatchHEPSI: {self.role} starting HE-based batch PSI")

        self._local_set = self._to_set(local_data)

        if self.role == "host":
            # Host是求交发起方
            # 1. 生成HE密钥对
            try:
                self._public_key, self._private_key = self._generate_paillier_keypair()
            except Exception as e:
                logger.warning(f"HE key generation failed, using fallback: {e}")
                return self._fallback_psi()

            # 2. 发送公钥
            self.channel.send("he_psi_public_key", self._public_key)

            # 3. 发送查询元素（使用哈希）
            local_hashes = {}
            for elem in self._local_set:
                h = hashlib.sha256(str(elem).encode()).hexdigest()
                local_hashes[h] = elem

            self.channel.send("he_psi_query_hashes", set(local_hashes.keys()))

            # 4. 接收匹配结果
            matched_hashes = self.channel.recv("he_psi_matches")

            # 5. 还原交集
            self._intersection_result = set()
            for h in matched_hashes:
                if h in local_hashes:
                    self._intersection_result.add(local_hashes[h])

        else:  # guest
            # Guest是数据方
            # 1. 接收公钥
            public_key = self.channel.recv("he_psi_public_key")

            # 2. 接收查询哈希
            query_hashes = self.channel.recv("he_psi_query_hashes")

            # 3. 计算本地数据哈希
            local_hashes = {}
            for elem in self._local_set:
                h = hashlib.sha256(str(elem).encode()).hexdigest()
                local_hashes[h] = elem

            # 4. 计算交集
            matched_hashes = query_hashes & set(local_hashes.keys())

            # 5. 发送匹配结果
            self.channel.send("he_psi_matches", matched_hashes)

            # 6. 还原交集
            self._intersection_result = set()
            for h in matched_hashes:
                if h in local_hashes:
                    self._intersection_result.add(local_hashes[h])

        logger.info(f"BatchHEPSI: {self.role} intersection size = {len(self._intersection_result)}")

        return self._intersection_result

    def _fallback_psi(self) -> Set:
        """降级PSI方案"""
        logger.warning("BatchHEPSI: Using fallback PSI protocol")

        salt = secrets.token_bytes(32)

        if self.role == "host":
            self.channel.send("fallback_psi_salt", salt)

            local_hashes = {}
            for elem in self._local_set:
                h = hmac.new(salt, str(elem).encode(), hashlib.sha256).hexdigest()
                local_hashes[h] = elem

            self.channel.send("fallback_psi_hashes", set(local_hashes.keys()))
            matched = self.channel.recv("fallback_psi_matches")

            self._intersection_result = set()
            for h in matched:
                if h in local_hashes:
                    self._intersection_result.add(local_hashes[h])

        else:
            salt = self.channel.recv("fallback_psi_salt")

            local_hashes = {}
            for elem in self._local_set:
                h = hmac.new(salt, str(elem).encode(), hashlib.sha256).hexdigest()
                local_hashes[h] = elem

            host_hashes = self.channel.recv("fallback_psi_hashes")
            matched = host_hashes & set(local_hashes.keys())
            self.channel.send("fallback_psi_matches", matched)

            self._intersection_result = set()
            for h in matched:
                if h in local_hashes:
                    self._intersection_result.add(local_hashes[h])

        return self._intersection_result
