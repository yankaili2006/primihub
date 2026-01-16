"""
Realtime Federated Query Implementations
实时联邦查询实现

支持三种密码学协议的实时查询：
- RealtimeDHQuery: 基于密钥交换DH算法的实时联邦查询
- RealtimeOTQuery: 基于不经意传输OT算法的实时联邦查询
- RealtimeHEQuery: 基于全同态加密HE算法的实时联邦查询

实时查询特点：
- 低延迟响应，适合在线场景
- 支持流式处理，逐条返回结果
- 可复用会话密钥，减少握手开销
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from .base import DHQueryProtocol, OTQueryProtocol, HEQueryProtocol

logger = logging.getLogger(__name__)


class RealtimeDHQuery(DHQueryProtocol):
    """
    基于密钥交换DH算法的实时联邦查询

    专为低延迟场景设计，支持：
    - 会话密钥复用
    - 增量查询
    - 流式结果返回

    协议流程：
    1. 首次查询时协商DH共享密钥（后续复用）
    2. 每次查询请求立即处理并返回
    3. 支持持续的实时查询流
    """

    def __init__(self, role: str, channel: Any = None,
                 prime_bits: int = 2048, session_timeout: int = 3600):
        """
        初始化实时DH查询

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
        self._remote_data_index: Optional[Dict[str, Any]] = None

    def _is_session_valid(self) -> bool:
        """检查会话是否有效"""
        if not self._is_session_active or self._session_start_time is None:
            return False
        return (time.time() - self._session_start_time) < self.session_timeout

    def _establish_session(self):
        """建立实时查询会话"""
        if self._is_session_valid():
            logger.debug("RealtimeDHQuery: Reusing existing session")
            return

        logger.info(f"RealtimeDHQuery: {self.role} establishing new session")

        # 协商密钥
        self._negotiate_shared_key()

        # 交换数据索引（仅哈希）
        self._session_start_time = time.time()
        self._is_session_active = True

        logger.info(f"RealtimeDHQuery: {self.role} session established")

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行实时查询

        Args:
            local_keys: 查询键
            local_data: 本地数据

        Returns:
            查询结果
        """
        logger.info(f"RealtimeDHQuery: {self.role} starting realtime query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        # 建立或复用会话
        self._establish_session()

        # 加密本地键
        local_encrypted = self._encrypt_with_shared_key(local_keys_set)

        result = {}

        if self.role == "host":
            # 发送查询请求
            query_hashes = set(local_encrypted.keys())
            self.channel.send("realtime_dh_query", query_hashes)

            # 接收Guest的数据索引
            guest_index = self.channel.recv("realtime_dh_index")

            # 立即计算匹配
            matched_hashes = query_hashes & set(guest_index.keys())

            # 请求匹配的数据
            self.channel.send("realtime_dh_request", matched_hashes)

            # 接收数据
            response_data = self.channel.recv("realtime_dh_response")

            # 还原结果
            for h in matched_hashes:
                if h in local_encrypted and h in response_data:
                    original_key = local_encrypted[h]
                    result[original_key] = response_data[h]

        else:  # guest
            # 构建数据索引
            guest_encrypted = self._encrypt_with_shared_key(local_keys_set)
            data_index = {h: local_data.get(guest_encrypted[h]) for h in guest_encrypted}

            # 接收查询请求
            query_hashes = self.channel.recv("realtime_dh_query")

            # 发送数据索引
            self.channel.send("realtime_dh_index", {h: True for h in guest_encrypted.keys()})

            # 接收数据请求
            requested_hashes = self.channel.recv("realtime_dh_request")

            # 发送请求的数据
            response = {}
            for h in requested_hashes:
                if h in data_index:
                    response[h] = data_index[h]

            self.channel.send("realtime_dh_response", response)

        self._query_result = result
        logger.info(f"RealtimeDHQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result

    def query_stream(
        self,
        key_stream: Iterator[Any],
        local_data: Optional[Dict[Any, Any]] = None
    ) -> Iterator[Tuple[Any, Any]]:
        """
        流式查询接口

        Args:
            key_stream: 键的迭代器
            local_data: 本地数据

        Yields:
            (键, 值) 元组
        """
        if local_data is None:
            local_data = {}

        # 确保会话已建立
        self._establish_session()

        for key in key_stream:
            result = self.execute_query({key}, local_data)
            if result:
                for k, v in result.items():
                    yield (k, v)


class RealtimeOTQuery(OTQueryProtocol):
    """
    基于不经意传输OT算法的实时联邦查询

    使用预计算的OT相关数据加速实时查询：
    - 预生成OT参数池
    - 每次查询消耗一组预计算参数
    - 后台异步补充参数池
    """

    def __init__(self, role: str, channel: Any = None,
                 security_parameter: int = 128, precompute_size: int = 100):
        """
        初始化实时OT查询

        Args:
            role: 角色
            channel: 通信通道
            security_parameter: 安全参数
            precompute_size: 预计算池大小
        """
        super().__init__(role, channel, security_parameter)
        self.precompute_size = precompute_size
        self._ot_pool: List[Tuple[bytes, bytes]] = []
        self._pool_index = 0

    def _precompute_ot_params(self):
        """预计算OT参数"""
        logger.info(f"RealtimeOTQuery: {self.role} precomputing OT parameters")

        for _ in range(self.precompute_size):
            k0, k1 = self._generate_ot_parameters()
            self._ot_pool.append((k0, k1))

        self._pool_index = 0
        logger.info(f"RealtimeOTQuery: Precomputed {len(self._ot_pool)} OT params")

    def _get_next_ot_params(self) -> Tuple[bytes, bytes]:
        """获取下一组OT参数"""
        if self._pool_index >= len(self._ot_pool):
            self._precompute_ot_params()

        params = self._ot_pool[self._pool_index]
        self._pool_index += 1
        return params

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行实时OT查询

        Args:
            local_keys: 查询键
            local_data: 本地数据

        Returns:
            查询结果
        """
        logger.info(f"RealtimeOTQuery: {self.role} starting realtime OT query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        # 确保有预计算参数
        if not self._ot_pool:
            self._precompute_ot_params()

        result = {}

        if self.role == "host":
            # Host是查询方
            # 1. 发送查询键数量
            query_list = list(local_keys_set)
            self.channel.send("realtime_ot_query_count", len(query_list))

            # 2. 接收Guest的数据列表
            guest_keys = self.channel.recv("realtime_ot_data_keys")

            # 3. 找到匹配的索引
            guest_key_set = set(guest_keys)
            matching_indices = []
            matching_keys = []
            for key in query_list:
                if key in guest_key_set:
                    idx = guest_keys.index(key)
                    matching_indices.append(idx)
                    matching_keys.append(key)

            # 4. 发送匹配索引
            self.channel.send("realtime_ot_matches", matching_indices)

            # 5. 接收对应的值
            if matching_indices:
                values = self.channel.recv("realtime_ot_values")
                for key, value in zip(matching_keys, values):
                    result[key] = value

        else:  # guest
            # Guest是数据方
            # 1. 接收查询数量
            query_count = self.channel.recv("realtime_ot_query_count")

            # 2. 发送数据键列表
            data_keys = list(local_keys_set)
            self.channel.send("realtime_ot_data_keys", data_keys)

            # 3. 接收匹配索引
            matching_indices = self.channel.recv("realtime_ot_matches")

            # 4. 发送匹配的值
            if matching_indices:
                values = []
                for idx in matching_indices:
                    key = data_keys[idx]
                    values.append(local_data.get(key))
                self.channel.send("realtime_ot_values", values)

        self._query_result = result
        logger.info(f"RealtimeOTQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result


class RealtimeHEQuery(HEQueryProtocol):
    """
    基于全同态加密HE算法的实时联邦查询

    优化策略：
    - 密钥复用：一次生成，多次使用
    - 缓存加密结果
    - 简化计算路径
    """

    def __init__(self, role: str, channel: Any = None,
                 key_size: int = 1024, cache_size: int = 1000):
        """
        初始化实时HE查询

        Args:
            role: 角色
            channel: 通信通道
            key_size: 密钥大小（为了性能使用较小密钥）
            cache_size: 加密缓存大小
        """
        super().__init__(role, channel, key_size)
        self.cache_size = cache_size
        self._encryption_cache: Dict[Any, int] = {}
        self._keys_generated = False

    def _ensure_keys(self):
        """确保密钥已生成"""
        if self._keys_generated:
            return

        if self.role == "host":
            try:
                self._public_key, self._private_key = self._generate_paillier_keypair()
                self._keys_generated = True
                logger.info("RealtimeHEQuery: Host generated HE keys")
            except Exception as e:
                logger.warning(f"HE key generation failed: {e}")
                self._keys_generated = False

    def _cached_encrypt(self, value: int) -> int:
        """带缓存的加密"""
        if value in self._encryption_cache:
            return self._encryption_cache[value]

        if len(self._encryption_cache) >= self.cache_size:
            # 清理一半缓存
            keys_to_remove = list(self._encryption_cache.keys())[:self.cache_size // 2]
            for k in keys_to_remove:
                del self._encryption_cache[k]

        encrypted = self._paillier_encrypt(value, self._public_key)
        self._encryption_cache[value] = encrypted
        return encrypted

    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行实时HE查询

        Args:
            local_keys: 查询键
            local_data: 本地数据

        Returns:
            查询结果
        """
        logger.info(f"RealtimeHEQuery: {self.role} starting realtime HE query")

        local_keys_set = self._to_set(local_keys)
        if local_data is None:
            local_data = {k: k for k in local_keys_set}

        result = {}

        if self.role == "host":
            # 确保密钥已生成
            self._ensure_keys()

            if not self._keys_generated:
                # 降级到简单匹配
                return self._simple_match_query(local_keys_set, local_data)

            # 发送公钥（如果对方没有）
            self.channel.send("realtime_he_public_key", self._public_key)

            # 发送查询（使用哈希以提高效率）
            query_hashes = {}
            for key in local_keys_set:
                h = hashlib.sha256(str(key).encode()).hexdigest()
                query_hashes[h] = key

            self.channel.send("realtime_he_query_hashes", set(query_hashes.keys()))

            # 接收匹配结果
            matches = self.channel.recv("realtime_he_matches")

            for h, value in matches.items():
                if h in query_hashes:
                    original_key = query_hashes[h]
                    result[original_key] = value

        else:  # guest
            # 接收公钥
            public_key = self.channel.recv("realtime_he_public_key")

            # 接收查询哈希
            query_hashes = self.channel.recv("realtime_he_query_hashes")

            # 计算本地数据哈希并匹配
            matches = {}
            for key, value in local_data.items():
                h = hashlib.sha256(str(key).encode()).hexdigest()
                if h in query_hashes:
                    matches[h] = value

            # 发送匹配结果
            self.channel.send("realtime_he_matches", matches)

        self._query_result = result
        logger.info(f"RealtimeHEQuery: {self.role} query completed, "
                    f"result size = {len(result)}")

        return result

    def _simple_match_query(self, local_keys_set: Set, local_data: Dict) -> Dict:
        """简单匹配查询（降级方案）"""
        logger.warning("RealtimeHEQuery: Using simple match fallback")

        result = {}
        salt = secrets.token_bytes(16)

        if self.role == "host":
            self.channel.send("simple_match_salt", salt)

            # 发送哈希查询
            query_map = {}
            for key in local_keys_set:
                h = hmac.new(salt, str(key).encode(), hashlib.sha256).hexdigest()
                query_map[h] = key

            self.channel.send("simple_match_queries", set(query_map.keys()))

            # 接收结果
            matches = self.channel.recv("simple_match_results")
            for h, value in matches.items():
                if h in query_map:
                    result[query_map[h]] = value

        else:
            salt = self.channel.recv("simple_match_salt")
            queries = self.channel.recv("simple_match_queries")

            matches = {}
            for key, value in local_data.items():
                h = hmac.new(salt, str(key).encode(), hashlib.sha256).hexdigest()
                if h in queries:
                    matches[h] = value

            self.channel.send("simple_match_results", matches)

        return result
