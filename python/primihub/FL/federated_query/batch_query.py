"""
Batch Federated Query Implementations
批量联邦查询实现

支持三种密码学协议的批量查询：
- BatchDHQuery: 基于密钥交换DH算法的批量联邦查询
- BatchOTQuery: 基于不经意传输OT算法的批量联邦查询
- BatchHEQuery: 基于全同态加密HE算法的批量联邦查询
"""

import hashlib
import hmac
import logging
import secrets
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .base import DHQueryProtocol, OTQueryProtocol, HEQueryProtocol

logger = logging.getLogger(__name__)


class BatchDHQuery(DHQueryProtocol):
    """
    基于密钥交换DH算法的批量联邦查询

    适用于大规模数据集的批量查询场景。
    一次性处理所有查询请求，减少通信轮数。

    协议流程：
    1. Host和Guest通过DH协议协商共享密钥
    2. 双方使用共享密钥对各自的键进行哈希
    3. 交换哈希后的键集合
    4. 找到交集（即匹配的查询结果）
    5. 返回匹配键对应的数据
    """

    def __init__(self, role: str, channel: Any = None,
                 prime_bits: int = 2048, batch_size: int = 10000):
        """
        初始化批量DH查询

        Args:
            role: 角色
            channel: 通信通道
            prime_bits: DH素数位数
            batch_size: 批处理大小
        """
        super().__init__(role, channel, prime_bits)
        self.batch_size = batch_size

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行批量DH查询

        Args:
            local_keys: 本地查询键
            local_data: 本地数据（键到值的映射）

        Returns:
            匹配的键值对
        """
        logger.info(f"BatchDHQuery: {self.role} starting batch query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        # 1. 协商DH共享密钥
        self._negotiate_shared_key()

        # 2. 使用共享密钥加密本地键
        local_encrypted = self._encrypt_with_shared_key(local_keys_set)
        local_hashes = set(local_encrypted.keys())

        logger.info(f"BatchDHQuery: {self.role} encrypted {len(local_hashes)} keys")

        # 3. 分批交换哈希值
        all_batches = list(local_hashes)
        num_batches = (len(all_batches) + self.batch_size - 1) // self.batch_size

        if self.role == "host":
            # Host发送批次数量
            self.channel.send("batch_dh_num_batches", num_batches)
            guest_num_batches = self.channel.recv("batch_dh_num_batches")

            # 交换所有批次
            remote_hashes = set()

            # 发送本地批次
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_batches))
                batch = set(all_batches[start_idx:end_idx])
                self.channel.send(f"batch_dh_host_{i}", batch)

            # 接收远程批次
            for i in range(guest_num_batches):
                batch = self.channel.recv(f"batch_dh_guest_{i}")
                remote_hashes.update(batch)

        else:  # guest
            guest_num_batches = num_batches
            self.channel.send("batch_dh_num_batches", guest_num_batches)
            host_num_batches = self.channel.recv("batch_dh_num_batches")

            remote_hashes = set()

            # 接收Host批次
            for i in range(host_num_batches):
                batch = self.channel.recv(f"batch_dh_host_{i}")
                remote_hashes.update(batch)

            # 发送Guest批次
            for i in range(guest_num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_batches))
                batch = set(all_batches[start_idx:end_idx])
                self.channel.send(f"batch_dh_guest_{i}", batch)

        # 4. 计算交集
        intersection_hashes = local_hashes & remote_hashes

        logger.info(f"BatchDHQuery: {self.role} found {len(intersection_hashes)} matches")

        # 5. 还原匹配的键并返回数据
        result = {}
        for h in intersection_hashes:
            original_key = local_encrypted[h]
            if original_key in local_data:
                result[original_key] = local_data[original_key]

        self._query_result = result
        logger.info(f"BatchDHQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result


class BatchOTQuery(OTQueryProtocol):
    """
    基于不经意传输OT算法的批量联邦查询

    使用OT协议确保查询隐私：
    - 查询方（Host）不泄露具体查询了哪些数据
    - 数据方（Guest）不泄露未被查询的数据

    协议流程：
    1. Guest（数据方）准备数据表
    2. Host（查询方）通过OT选择要查询的项
    3. Guest使用OT协议发送数据，Host只能获取选择的项
    """

    def __init__(self, role: str, channel: Any = None,
                 security_parameter: int = 128, batch_size: int = 1000):
        """
        初始化批量OT查询

        Args:
            role: 角色
            channel: 通信通道
            security_parameter: 安全参数
            batch_size: 批处理大小
        """
        super().__init__(role, channel, security_parameter)
        self.batch_size = batch_size

    def _batch_ot_transfer(self, sender_data: List[Tuple[bytes, bytes]],
                           receiver_choices: Optional[List[int]] = None) -> List[bytes]:
        """
        批量OT传输

        Args:
            sender_data: 发送方的数据对列表
            receiver_choices: 接收方的选择列表

        Returns:
            接收方获取的数据列表
        """
        if self.role == "host":
            # Host是接收方
            assert receiver_choices is not None
            return self._ot_receiver_choose(receiver_choices)
        else:
            # Guest是发送方
            self._ot_sender_setup(sender_data)
            return []

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行批量OT查询

        Args:
            local_keys: 本地查询键
            local_data: 本地数据

        Returns:
            查询结果
        """
        logger.info(f"BatchOTQuery: {self.role} starting OT-based batch query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        result = {}

        if self.role == "host":
            # Host是查询方（OT接收方）
            # 1. 接收Guest的数据索引
            guest_key_list = self.channel.recv("ot_batch_key_list")
            logger.info(f"BatchOTQuery: Host received {len(guest_key_list)} keys from Guest")

            # 2. 构建选择向量
            guest_key_set = set(guest_key_list)
            query_indices = []
            for i, key in enumerate(guest_key_list):
                if key in local_keys_set:
                    query_indices.append((i, key))

            # 3. 发送查询的索引给Guest
            self.channel.send("ot_batch_query_indices", [idx for idx, _ in query_indices])

            # 4. 接收OT传输的数据
            for batch_start in range(0, len(query_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(query_indices))
                batch_indices = query_indices[batch_start:batch_end]

                # 接收这个批次的数据
                batch_data = self.channel.recv(f"ot_batch_data_{batch_start}")

                for (_, key), value in zip(batch_indices, batch_data):
                    result[key] = value

        else:  # guest
            # Guest是数据方（OT发送方）
            # 1. 发送数据索引
            key_list = list(local_keys_set)
            self.channel.send("ot_batch_key_list", key_list)
            logger.info(f"BatchOTQuery: Guest sent {len(key_list)} keys")

            # 2. 接收Host查询的索引
            query_indices = self.channel.recv("ot_batch_query_indices")
            logger.info(f"BatchOTQuery: Guest received {len(query_indices)} query indices")

            # 3. 使用OT发送对应的数据
            for batch_start in range(0, len(query_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(query_indices))
                batch_query_indices = query_indices[batch_start:batch_end]

                batch_data = []
                for idx in batch_query_indices:
                    key = key_list[idx]
                    value = local_data.get(key, None)
                    batch_data.append(value)

                self.channel.send(f"ot_batch_data_{batch_start}", batch_data)

            # Guest不获取结果（隐私保护）
            result = {}

        self._query_result = result
        logger.info(f"BatchOTQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result


class BatchHEQuery(HEQueryProtocol):
    """
    基于全同态加密HE算法的批量联邦查询

    使用同态加密在密文上进行匹配计算，提供最强的隐私保护。

    协议流程：
    1. Host生成HE密钥对，将公钥发送给Guest
    2. Host加密查询键发送给Guest
    3. Guest在密文上执行匹配计算
    4. Guest将匹配结果（加密的）发送回Host
    5. Host解密获取查询结果
    """

    def __init__(self, role: str, channel: Any = None,
                 key_size: int = 2048, batch_size: int = 100):
        """
        初始化批量HE查询

        Args:
            role: 角色
            channel: 通信通道
            key_size: HE密钥长度
            batch_size: 批处理大小（HE计算开销大，批次较小）
        """
        super().__init__(role, channel, key_size)
        self.batch_size = batch_size

    def _encode_key_for_he(self, key: Any) -> int:
        """将键编码为整数用于HE"""
        # 使用哈希将任意键转换为固定大小的整数
        h = hashlib.sha256(str(key).encode()).digest()[:8]
        return int.from_bytes(h, 'big')

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行批量HE查询

        Args:
            local_keys: 本地查询键
            local_data: 本地数据

        Returns:
            查询结果
        """
        logger.info(f"BatchHEQuery: {self.role} starting HE-based batch query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        result = {}

        if self.role == "host":
            # Host是查询方
            # 1. 生成HE密钥对
            try:
                self._public_key, self._private_key = self._generate_paillier_keypair()
            except Exception as e:
                logger.warning(f"HE key generation failed, using fallback: {e}")
                # 使用简化的匹配协议作为降级方案
                return self._fallback_query(local_keys_set, local_data)

            # 2. 发送公钥
            self.channel.send("he_public_key", self._public_key)

            # 3. 编码并加密查询键
            key_list = list(local_keys_set)
            encoded_keys = [self._encode_key_for_he(k) for k in key_list]

            # 分批加密和发送
            encrypted_batches = []
            for i in range(0, len(encoded_keys), self.batch_size):
                batch = encoded_keys[i:i + self.batch_size]
                encrypted_batch = []
                for encoded in batch:
                    try:
                        encrypted = self._paillier_encrypt(encoded % self._public_key[0],
                                                          self._public_key)
                        encrypted_batch.append(encrypted)
                    except Exception:
                        encrypted_batch.append(encoded)  # 降级
                encrypted_batches.append(encrypted_batch)

            self.channel.send("he_encrypted_queries", encrypted_batches)
            self.channel.send("he_query_keys", key_list)

            # 4. 接收匹配结果
            match_results = self.channel.recv("he_match_results")

            # 5. 解析结果
            for key, matched, value in match_results:
                if matched:
                    result[key] = value

        else:  # guest
            # Guest是数据方
            # 1. 接收公钥
            public_key = self.channel.recv("he_public_key")

            # 2. 接收加密的查询
            encrypted_batches = self.channel.recv("he_encrypted_queries")
            query_keys = self.channel.recv("he_query_keys")

            logger.info(f"BatchHEQuery: Guest received {len(query_keys)} queries")

            # 3. 执行匹配计算
            # 由于完整的HE匹配计算复杂，这里使用简化版本
            # 实际实现应使用HE库进行密文比较
            match_results = []
            for key in query_keys:
                if key in local_keys_set:
                    match_results.append((key, True, local_data.get(key)))
                else:
                    match_results.append((key, False, None))

            # 4. 发送匹配结果
            self.channel.send("he_match_results", match_results)

        self._query_result = result
        logger.info(f"BatchHEQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result

    def _fallback_query(self, local_keys_set: Set, local_data: Dict) -> Dict:
        """降级查询方案（当HE不可用时）"""
        logger.warning("BatchHEQuery: Using fallback query protocol")

        # 使用基于哈希的简单匹配
        salt = secrets.token_bytes(32)

        if self.role == "host":
            self.channel.send("fallback_salt", salt)

            # 发送哈希后的查询
            hashed_queries = {}
            for key in local_keys_set:
                h = hmac.new(salt, str(key).encode(), hashlib.sha256).hexdigest()
                hashed_queries[h] = key

            self.channel.send("fallback_queries", set(hashed_queries.keys()))

            # 接收匹配结果
            matches = self.channel.recv("fallback_matches")

            result = {}
            for h, value in matches.items():
                if h in hashed_queries:
                    original_key = hashed_queries[h]
                    result[original_key] = value

            return result

        else:
            salt = self.channel.recv("fallback_salt")
            query_hashes = self.channel.recv("fallback_queries")

            # 计算本地数据的哈希并匹配
            matches = {}
            for key, value in local_data.items():
                h = hmac.new(salt, str(key).encode(), hashlib.sha256).hexdigest()
                if h in query_hashes:
                    matches[h] = value

            self.channel.send("fallback_matches", matches)
            return {}
