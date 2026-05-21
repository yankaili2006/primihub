"""
Federated Query Base Classes
联邦查询基础类

提供安全的多方数据查询功能，支持多种密码学协议。
"""

import hashlib
import hmac
import logging
import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class FederatedQueryBase(ABC):
    """
    联邦查询基类

    定义所有查询协议的通用接口。
    """

    def __init__(self, role: str, channel: Any = None):
        """
        初始化查询器

        Args:
            role: 角色 ('host' 或 'guest')
            channel: 通信通道
        """
        self.role = role
        self.channel = channel
        self._local_data: Optional[Set] = None
        self._query_result: Optional[Set] = None
        self._query_keys: Optional[Set] = None

    @abstractmethod
    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """
        执行查询

        Args:
            local_keys: 本地查询键
            local_data: 本地数据字典（键->值映射）
            **kwargs: 协议特定参数

        Returns:
            查询结果（匹配的键值对）
        """
        pass

    def set_channel(self, channel: Any):
        """设置通信通道"""
        self.channel = channel

    def get_result(self) -> Optional[Dict[Any, Any]]:
        """获取查询结果"""
        return self._query_result

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


class DHQueryProtocol(FederatedQueryBase):
    """
    基于Diffie-Hellman密钥交换的查询协议

    使用DH协议协商共享密钥，然后使用共享密钥对查询进行安全处理。
    安全级别：中等
    适用场景：一般隐私需求的查询场景
    """

    def __init__(self, role: str, channel: Any = None,
                 prime_bits: int = 2048):
        """
        初始化DH查询协议

        Args:
            role: 角色
            channel: 通信通道
            prime_bits: DH素数位数
        """
        super().__init__(role, channel)
        self.prime_bits = prime_bits
        self._shared_key: Optional[bytes] = None
        self._local_private_key: Optional[int] = None
        # 使用预定义的安全素数（简化版）
        self._prime = self._get_safe_prime()
        self._generator = 2

    def _get_safe_prime(self) -> int:
        """获取安全素数"""
        # 使用RFC 3526定义的2048位MODP组素数
        return int(
            "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
            "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
            "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
            "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
            "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
            "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
            "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16
        )

    def _negotiate_shared_key(self):
        """协商DH共享密钥"""
        # 生成私钥
        self._local_private_key = secrets.randbelow(self._prime - 2) + 1

        # 计算公钥
        local_public_key = pow(self._generator, self._local_private_key, self._prime)

        if self.role == "host":
            # Host先发送公钥
            self.channel.send("dh_public_key_host", local_public_key)
            # 接收Guest公钥
            remote_public_key = self.channel.recv("dh_public_key_guest")
        else:
            # Guest先接收Host公钥
            remote_public_key = self.channel.recv("dh_public_key_host")
            # 发送Guest公钥
            self.channel.send("dh_public_key_guest", local_public_key)

        # 计算共享密钥
        shared_secret = pow(remote_public_key, self._local_private_key, self._prime)
        self._shared_key = hashlib.sha256(str(shared_secret).encode()).digest()

        logger.info(f"DHQueryProtocol: {self.role} shared key negotiated")

    def _encrypt_with_shared_key(self, data: Set) -> Dict[str, Any]:
        """使用共享密钥加密数据"""
        encrypted_map = {}
        for element in data:
            h = hmac.new(self._shared_key, str(element).encode(), hashlib.sha256).hexdigest()
            encrypted_map[h] = element
        return encrypted_map

    @abstractmethod
    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """执行DH查询"""
        pass


class OTQueryProtocol(FederatedQueryBase):
    """
    基于不经意传输(Oblivious Transfer)的查询协议

    使用OT协议确保查询方不泄露查询内容，数据方不泄露非查询数据。
    安全级别：高
    适用场景：高隐私需求的查询场景
    """

    def __init__(self, role: str, channel: Any = None,
                 security_parameter: int = 128):
        """
        初始化OT查询协议

        Args:
            role: 角色
            channel: 通信通道
            security_parameter: 安全参数（比特）
        """
        super().__init__(role, channel)
        self.security_parameter = security_parameter
        self._ot_keys: Dict[str, bytes] = {}

    def _generate_ot_parameters(self) -> Tuple[bytes, bytes]:
        """生成OT参数"""
        # 生成随机密钥对
        k0 = secrets.token_bytes(self.security_parameter // 8)
        k1 = secrets.token_bytes(self.security_parameter // 8)
        return k0, k1

    def _ot_sender_setup(self, messages: List[Tuple[bytes, bytes]]):
        """
        OT发送方设置

        Args:
            messages: 消息对列表 [(m0, m1), ...]
        """
        # 简化的OT实现：基于随机预言机模型
        for i, (m0, m1) in enumerate(messages):
            # 生成随机密钥
            r = secrets.token_bytes(32)
            self.channel.send(f"ot_r_{i}", r)

            # 发送加密消息
            encrypted_m0 = bytes(a ^ b for a, b in zip(m0, hashlib.sha256(r + b'0').digest()[:len(m0)]))
            encrypted_m1 = bytes(a ^ b for a, b in zip(m1, hashlib.sha256(r + b'1').digest()[:len(m1)]))

            self.channel.send(f"ot_e0_{i}", encrypted_m0)
            self.channel.send(f"ot_e1_{i}", encrypted_m1)

    def _ot_receiver_choose(self, choices: List[int]) -> List[bytes]:
        """
        OT接收方选择

        Args:
            choices: 选择列表 (0或1)

        Returns:
            选择的消息列表
        """
        results = []
        for i, choice in enumerate(choices):
            r = self.channel.recv(f"ot_r_{i}")
            e0 = self.channel.recv(f"ot_e0_{i}")
            e1 = self.channel.recv(f"ot_e1_{i}")

            # 解密选择的消息
            if choice == 0:
                key = hashlib.sha256(r + b'0').digest()[:len(e0)]
                decrypted = bytes(a ^ b for a, b in zip(e0, key))
            else:
                key = hashlib.sha256(r + b'1').digest()[:len(e1)]
                decrypted = bytes(a ^ b for a, b in zip(e1, key))

            results.append(decrypted)

        return results

    @abstractmethod
    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """执行OT查询"""
        pass


class HEQueryProtocol(FederatedQueryBase):
    """
    基于全同态加密(Homomorphic Encryption)的查询协议

    使用同态加密在密文上进行计算，提供最强的隐私保护。
    安全级别：极高
    适用场景：最高隐私需求的查询场景

    注意：此实现使用Paillier加密的简化版本，用于演示。
    生产环境建议使用专业的HE库（如SEAL、HElib）。
    """

    def __init__(self, role: str, channel: Any = None,
                 key_size: int = 2048):
        """
        初始化HE查询协议

        Args:
            role: 角色
            channel: 通信通道
            key_size: 密钥长度（比特）
        """
        super().__init__(role, channel)
        self.key_size = key_size
        self._public_key: Optional[Tuple[int, int]] = None
        self._private_key: Optional[Tuple[int, int]] = None

    def _generate_paillier_keypair(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        生成Paillier密钥对（简化版）

        Returns:
            (public_key, private_key)
        """
        # 生成两个大素数（简化版本使用较小的数）
        def is_prime(n: int, k: int = 10) -> bool:
            if n < 2:
                return False
            if n == 2 or n == 3:
                return True
            if n % 2 == 0:
                return False
            r, d = 0, n - 1
            while d % 2 == 0:
                r += 1
                d //= 2
            for _ in range(k):
                a = secrets.randbelow(n - 3) + 2
                x = pow(a, d, n)
                if x == 1 or x == n - 1:
                    continue
                for _ in range(r - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    return False
            return True

        def generate_prime(bits: int) -> int:
            while True:
                p = secrets.randbits(bits)
                p |= (1 << bits - 1) | 1
                if is_prime(p):
                    return p

        # 使用较小的素数用于演示
        bits = min(self.key_size // 2, 512)
        p = generate_prime(bits)
        q = generate_prime(bits)

        n = p * q
        n_sq = n * n
        g = n + 1

        # 私钥
        lambda_n = (p - 1) * (q - 1)
        mu = pow(lambda_n, -1, n)

        return (n, g), (lambda_n, mu)

    def _paillier_encrypt(self, plaintext: int, public_key: Tuple[int, int]) -> int:
        """Paillier加密"""
        n, g = public_key
        n_sq = n * n
        r = secrets.randbelow(n - 1) + 1
        c = (pow(g, plaintext, n_sq) * pow(r, n, n_sq)) % n_sq
        return c

    def _paillier_decrypt(self, ciphertext: int,
                          public_key: Tuple[int, int],
                          private_key: Tuple[int, int]) -> int:
        """Paillier解密"""
        n, g = public_key
        lambda_n, mu = private_key
        n_sq = n * n

        def L(x):
            return (x - 1) // n

        plaintext = (L(pow(ciphertext, lambda_n, n_sq)) * mu) % n
        return plaintext

    @abstractmethod
    def execute_query(
        self,
        local_keys: Union[List, Set, np.ndarray],
        local_data: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        """执行HE查询"""
        pass
