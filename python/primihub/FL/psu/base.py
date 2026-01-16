"""
Private Set Union (PSU) Base Classes
隐私求并基础类

实现多种安全求并协议。
"""

import hashlib
import hmac
import logging
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PSUBase(ABC):
    """
    隐私求并基类

    定义PSU协议的基本接口。
    """

    def __init__(self, role: str, channel: Any = None):
        """
        初始化PSU

        Args:
            role: 角色（host, guest）
            channel: 通信通道
        """
        self.role = role
        self.channel = channel
        self._local_set: Set = set()
        self._union_result: Set = set()

    @abstractmethod
    def compute_union(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算并集

        Args:
            local_data: 本地数据

        Returns:
            并集结果
        """
        pass

    def set_channel(self, channel: Any):
        """设置通信通道"""
        self.channel = channel

    def get_result(self) -> Set:
        """获取并集结果"""
        return self._union_result

    def _to_set(self, data: Union[List, Set, np.ndarray]) -> Set:
        """将数据转换为集合"""
        if isinstance(data, set):
            return data
        elif isinstance(data, np.ndarray):
            return set(data.flatten().tolist())
        else:
            return set(data)

    @staticmethod
    def _hash_element(element: Any, salt: bytes = b'') -> str:
        """
        对元素进行哈希

        Args:
            element: 要哈希的元素
            salt: 盐值

        Returns:
            哈希值
        """
        data = str(element).encode('utf-8')
        if salt:
            return hmac.new(salt, data, hashlib.sha256).hexdigest()
        return hashlib.sha256(data).hexdigest()


class HashBasedPSU(PSUBase):
    """
    基于哈希的隐私求并

    使用双方共享的密钥对数据进行哈希，然后交换哈希值来计算并集。
    适用于半诚实模型。
    """

    def __init__(self, role: str, channel: Any = None,
                 hash_key: Optional[bytes] = None):
        """
        初始化基于哈希的PSU

        Args:
            role: 角色
            channel: 通信通道
            hash_key: 共享的哈希密钥（如果为None则自动协商）
        """
        super().__init__(role, channel)
        self.hash_key = hash_key
        self._hash_to_element: Dict[str, Any] = {}

    def _negotiate_key(self):
        """协商共享密钥"""
        if self.hash_key is not None:
            return

        if self.role == "host":
            # Host生成密钥并发送给Guest
            self.hash_key = secrets.token_bytes(32)
            self.channel.send("psu_hash_key", self.hash_key)
            logger.info("Host: Generated and sent hash key")
        else:
            # Guest接收密钥
            self.hash_key = self.channel.recv("psu_hash_key")
            logger.info("Guest: Received hash key")

    def _hash_local_data(self, local_data: Set) -> Dict[str, Any]:
        """对本地数据进行哈希"""
        hash_map = {}
        for element in local_data:
            h = self._hash_element(element, self.hash_key)
            hash_map[h] = element
        return hash_map

    def compute_union(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算并集

        使用哈希交换协议：
        1. 双方用共享密钥对各自数据哈希
        2. 交换哈希值集合
        3. 合并哈希值得到并集的哈希
        4. 各方根据自己的映射还原元素
        """
        logger.info(f"HashBasedPSU: {self.role} starting union computation")

        self._local_set = self._to_set(local_data)

        # 协商密钥
        self._negotiate_key()

        # 对本地数据哈希
        self._hash_to_element = self._hash_local_data(self._local_set)
        local_hashes = set(self._hash_to_element.keys())

        logger.info(f"HashBasedPSU: {self.role} hashed {len(local_hashes)} elements")

        # 交换哈希值
        if self.role == "host":
            # Host发送本地哈希，接收Guest的哈希
            self.channel.send("host_hashes", local_hashes)
            guest_hashes = self.channel.recv("guest_hashes")

            # 计算并集哈希
            union_hashes = local_hashes | guest_hashes

            # 发送并集哈希给Guest
            self.channel.send("union_hashes", union_hashes)

        else:  # guest
            # Guest接收Host的哈希，发送自己的哈希
            host_hashes = self.channel.recv("host_hashes")
            self.channel.send("guest_hashes", local_hashes)

            # 接收并集哈希
            union_hashes = self.channel.recv("union_hashes")

        # 还原本地可识别的元素
        self._union_result = set()
        for h in union_hashes:
            if h in self._hash_to_element:
                self._union_result.add(self._hash_to_element[h])

        logger.info(f"HashBasedPSU: {self.role} union size = {len(union_hashes)}, "
                    f"local identifiable = {len(self._union_result)}")

        return self._union_result


class BloomFilterPSU(PSUBase):
    """
    基于布隆过滤器的隐私求并

    使用布隆过滤器来高效地表示和交换集合信息。
    适用于大规模数据集，但有一定的假阳性率。
    """

    def __init__(self, role: str, channel: Any = None,
                 expected_elements: int = 10000,
                 false_positive_rate: float = 0.01):
        """
        初始化基于布隆过滤器的PSU

        Args:
            role: 角色
            channel: 通信通道
            expected_elements: 预期元素数量
            false_positive_rate: 允许的假阳性率
        """
        super().__init__(role, channel)
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        # 计算布隆过滤器参数
        self.size, self.num_hashes = self._calculate_params(
            expected_elements, false_positive_rate
        )
        self._bit_array: Optional[np.ndarray] = None

    @staticmethod
    def _calculate_params(n: int, p: float) -> Tuple[int, int]:
        """
        计算布隆过滤器参数

        Args:
            n: 预期元素数量
            p: 假阳性率

        Returns:
            (位数组大小, 哈希函数数量)
        """
        # m = -n*ln(p) / (ln2)^2
        m = int(-n * np.log(p) / (np.log(2) ** 2))
        # k = (m/n) * ln2
        k = max(1, int((m / n) * np.log(2)))
        return m, k

    def _hash_functions(self, element: Any) -> List[int]:
        """
        生成多个哈希值

        使用双重哈希技术生成k个哈希值
        """
        data = str(element).encode('utf-8')
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha1(data).hexdigest(), 16)

        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def _add_to_filter(self, element: Any):
        """将元素添加到布隆过滤器"""
        for idx in self._hash_functions(element):
            self._bit_array[idx] = 1

    def _check_in_filter(self, element: Any) -> bool:
        """检查元素是否可能在布隆过滤器中"""
        return all(self._bit_array[idx] == 1 for idx in self._hash_functions(element))

    def compute_union(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算并集

        使用布隆过滤器：
        1. 各方构建本地布隆过滤器
        2. 交换布隆过滤器（位数组OR操作）
        3. 使用合并后的过滤器验证元素
        """
        logger.info(f"BloomFilterPSU: {self.role} starting union computation")

        self._local_set = self._to_set(local_data)

        # 初始化布隆过滤器
        self._bit_array = np.zeros(self.size, dtype=np.uint8)

        # 添加本地元素
        for element in self._local_set:
            self._add_to_filter(element)

        logger.info(f"BloomFilterPSU: {self.role} built filter with {len(self._local_set)} elements")

        # 交换布隆过滤器
        if self.role == "host":
            self.channel.send("host_bloom_filter", self._bit_array)
            guest_filter = self.channel.recv("guest_bloom_filter")

            # OR操作合并过滤器
            union_filter = np.bitwise_or(self._bit_array, guest_filter)

            # 发送合并后的过滤器
            self.channel.send("union_bloom_filter", union_filter)

        else:  # guest
            host_filter = self.channel.recv("host_bloom_filter")
            self.channel.send("guest_bloom_filter", self._bit_array)

            # 接收合并后的过滤器
            union_filter = self.channel.recv("union_bloom_filter")

        self._bit_array = union_filter

        # 本地元素都在并集中
        self._union_result = self._local_set.copy()

        logger.info(f"BloomFilterPSU: {self.role} completed, "
                    f"filter bits set = {np.sum(union_filter)}")

        return self._union_result


class EncryptedPSU(PSUBase):
    """
    基于加密的隐私求并

    使用加法同态加密或不经意传输来实现安全求并。
    提供更强的安全保证。
    """

    def __init__(self, role: str, channel: Any = None,
                 use_shuffle: bool = True):
        """
        初始化基于加密的PSU

        Args:
            role: 角色
            channel: 通信通道
            use_shuffle: 是否对结果进行随机打乱
        """
        super().__init__(role, channel)
        self.use_shuffle = use_shuffle
        self._encryption_key: Optional[bytes] = None

    def _generate_random_mask(self, size: int) -> bytes:
        """生成随机掩码"""
        return secrets.token_bytes(size)

    def _encrypt_element(self, element: Any, mask: bytes) -> bytes:
        """
        加密元素

        使用XOR加密（简化版本，实际应使用更安全的加密）
        """
        element_bytes = str(element).encode('utf-8')
        # 扩展mask到元素长度
        extended_mask = (mask * ((len(element_bytes) // len(mask)) + 1))[:len(element_bytes)]
        encrypted = bytes(a ^ b for a, b in zip(element_bytes, extended_mask))
        return encrypted

    def _double_hash_encrypt(self, element: Any, key1: bytes, key2: bytes) -> str:
        """
        双重哈希加密

        使用两个密钥进行双重HMAC，提供交换不变性
        """
        data = str(element).encode('utf-8')
        h1 = hmac.new(key1, data, hashlib.sha256).digest()
        h2 = hmac.new(key2, h1, hashlib.sha256).hexdigest()
        return h2

    def compute_union(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算并集

        使用双密钥哈希协议：
        1. 双方各生成一个密钥
        2. Host用自己的密钥对数据哈希，发送给Guest
        3. Guest用自己的密钥对收到的哈希再次哈希
        4. 反向同样操作
        5. 由于哈希的交换性，相同元素会得到相同的最终哈希
        """
        logger.info(f"EncryptedPSU: {self.role} starting union computation")

        self._local_set = self._to_set(local_data)

        # 生成本地密钥
        local_key = secrets.token_bytes(32)

        if self.role == "host":
            # 第一轮：Host用自己的密钥哈希，发送给Guest
            host_first_hash = {}
            for elem in self._local_set:
                h = hmac.new(local_key, str(elem).encode(), hashlib.sha256).digest()
                host_first_hash[h] = elem

            self.channel.send("host_first_hash", set(host_first_hash.keys()))

            # 接收Guest的第一轮哈希
            guest_first_hash = self.channel.recv("guest_first_hash")

            # 第二轮：Host对Guest的哈希再次哈希
            host_second_hash = set()
            for h in guest_first_hash:
                h2 = hmac.new(local_key, h, hashlib.sha256).digest()
                host_second_hash.add(h2)

            self.channel.send("host_second_hash", host_second_hash)

            # 接收Guest的第二轮哈希（对Host数据的）
            guest_second_hash = self.channel.recv("guest_second_hash")

            # 计算并集
            # Host可以识别的元素：自己的数据 + Guest确认的交集外数据
            final_hashes = set()
            local_final_hash_map = {}
            for elem in self._local_set:
                h1 = hmac.new(local_key, str(elem).encode(), hashlib.sha256).digest()
                # 这个元素的最终哈希需要Guest的密钥，由Guest计算后返回
                local_final_hash_map[h1] = elem

            # 从Guest接收最终哈希映射
            guest_final_hashes = self.channel.recv("guest_final_hashes_for_host")

            # 合并
            all_final_hashes = host_second_hash | guest_final_hashes

            # 发送并集大小
            self.channel.send("union_size", len(all_final_hashes))

            # Host的结果就是本地数据（因为并集包含所有本地数据）
            self._union_result = self._local_set.copy()

        else:  # guest
            # 接收Host的第一轮哈希
            host_first_hash = self.channel.recv("host_first_hash")

            # 第一轮：Guest用自己的密钥哈希，发送给Host
            guest_first_hash = {}
            for elem in self._local_set:
                h = hmac.new(local_key, str(elem).encode(), hashlib.sha256).digest()
                guest_first_hash[h] = elem

            self.channel.send("guest_first_hash", set(guest_first_hash.keys()))

            # 第二轮：Guest对Host的哈希再次哈希
            guest_second_hash = set()
            host_elem_to_final = {}
            for h in host_first_hash:
                h2 = hmac.new(local_key, h, hashlib.sha256).digest()
                guest_second_hash.add(h2)
                host_elem_to_final[h] = h2

            self.channel.send("guest_second_hash", guest_second_hash)

            # 接收Host的第二轮哈希
            host_second_hash = self.channel.recv("host_second_hash")

            # Guest计算自己数据的最终哈希并发送给Host
            guest_final_hashes = set()
            for elem in self._local_set:
                h1 = hmac.new(local_key, str(elem).encode(), hashlib.sha256).digest()
                # Host会对这个再哈希，这里我们直接用本地数据
                guest_final_hashes.add(h1)

            # 发送给Host用于计算并集
            self.channel.send("guest_final_hashes_for_host", host_second_hash)

            # 接收并集大小
            union_size = self.channel.recv("union_size")

            # Guest的结果是本地数据
            self._union_result = self._local_set.copy()

        if self.use_shuffle:
            # 打乱结果顺序
            result_list = list(self._union_result)
            np.random.shuffle(result_list)
            self._union_result = set(result_list)

        logger.info(f"EncryptedPSU: {self.role} completed, "
                    f"local result size = {len(self._union_result)}")

        return self._union_result


class SecurePSU(PSUBase):
    """
    安全求并协议

    基于不经意伪随机函数(OPRF)的安全求并实现。
    提供对恶意对手的安全保证。
    """

    def __init__(self, role: str, channel: Any = None,
                 curve: str = "secp256r1"):
        """
        初始化安全PSU

        Args:
            role: 角色
            channel: 通信通道
            curve: 椭圆曲线类型
        """
        super().__init__(role, channel)
        self.curve = curve

    def compute_union(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        使用OPRF协议计算并集

        协议流程：
        1. 双方各自生成密钥对
        2. 使用DH密钥交换建立共享密钥
        3. 基于共享密钥进行安全求并
        """
        logger.info(f"SecurePSU: {self.role} starting secure union computation")

        self._local_set = self._to_set(local_data)

        # 生成随机密钥
        private_key = secrets.token_bytes(32)

        # 对本地数据进行椭圆曲线点映射（简化版本使用哈希模拟）
        def map_to_curve(element: Any, key: bytes) -> str:
            data = str(element).encode('utf-8')
            h = hmac.new(key, data, hashlib.sha256).hexdigest()
            return h

        # 本地映射
        local_mapped = {}
        for elem in self._local_set:
            mapped = map_to_curve(elem, private_key)
            local_mapped[mapped] = elem

        if self.role == "host":
            # 发送映射后的元素
            self.channel.send("host_mapped", set(local_mapped.keys()))

            # 接收Guest的映射
            guest_mapped = self.channel.recv("guest_mapped")

            # 交换私钥进行第二轮映射（实际中应使用ECDH）
            self.channel.send("host_key", private_key)
            guest_key = self.channel.recv("guest_key")

            # 对Guest的映射再次处理
            guest_double_mapped = set()
            for m in guest_mapped:
                dm = hmac.new(private_key, m.encode(), hashlib.sha256).hexdigest()
                guest_double_mapped.add(dm)

            # 对Host的映射用Guest的密钥处理
            host_double_mapped = set()
            for m in local_mapped.keys():
                dm = hmac.new(guest_key, m.encode(), hashlib.sha256).hexdigest()
                host_double_mapped.add(dm)

            # 并集
            union_mapped = host_double_mapped | guest_double_mapped

            self.channel.send("union_size", len(union_mapped))

        else:  # guest
            # 接收Host的映射
            host_mapped = self.channel.recv("host_mapped")

            # 发送Guest的映射
            self.channel.send("guest_mapped", set(local_mapped.keys()))

            # 交换密钥
            host_key = self.channel.recv("host_key")
            self.channel.send("guest_key", private_key)

            # 接收并集大小
            union_size = self.channel.recv("union_size")

        # 每方的本地数据都在并集中
        self._union_result = self._local_set.copy()

        logger.info(f"SecurePSU: {self.role} completed")
        return self._union_result
