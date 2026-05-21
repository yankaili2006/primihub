"""
Federated PSI (Private Set Intersection) Base Classes
联邦求交基础类

提供安全的多方集合交集计算功能。
"""

import hashlib
import hmac
import logging
import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PSIBase(ABC):
    """
    联邦求交基类

    定义所有PSI协议的通用接口。
    """

    def __init__(self, role: str, channel: Any = None):
        """
        初始化PSI

        Args:
            role: 角色（host, guest）
            channel: 通信通道
        """
        self.role = role
        self.channel = channel
        self._local_set: Set = set()
        self._intersection_result: Set = set()

    @abstractmethod
    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """
        计算交集

        Args:
            local_data: 本地数据

        Returns:
            交集结果
        """
        pass

    def set_channel(self, channel: Any):
        """设置通信通道"""
        self.channel = channel

    def get_result(self) -> Set:
        """获取交集结果"""
        return self._intersection_result

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


class DHPSIProtocol(PSIBase):
    """
    基于Diffie-Hellman密钥交换的PSI协议

    使用DH协议协商共享密钥，然后使用共享密钥对数据进行加密比较。
    """

    def __init__(self, role: str, channel: Any = None, prime_bits: int = 2048):
        """
        初始化DH PSI协议

        Args:
            role: 角色
            channel: 通信通道
            prime_bits: DH素数位数
        """
        super().__init__(role, channel)
        self.prime_bits = prime_bits
        self._shared_key: Optional[bytes] = None
        self._prime = self._get_safe_prime()
        self._generator = 2

    def _get_safe_prime(self) -> int:
        """获取安全素数"""
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
        private_key = secrets.randbelow(self._prime - 2) + 1
        local_public_key = pow(self._generator, private_key, self._prime)

        if self.role == "host":
            self.channel.send("psi_dh_public_key_host", local_public_key)
            remote_public_key = self.channel.recv("psi_dh_public_key_guest")
        else:
            remote_public_key = self.channel.recv("psi_dh_public_key_host")
            self.channel.send("psi_dh_public_key_guest", local_public_key)

        shared_secret = pow(remote_public_key, private_key, self._prime)
        self._shared_key = hashlib.sha256(str(shared_secret).encode()).digest()

        logger.info(f"DHPSIProtocol: {self.role} shared key negotiated")

    def _encrypt_with_shared_key(self, data: Set) -> Dict[str, Any]:
        """使用共享密钥加密数据"""
        encrypted_map = {}
        for element in data:
            h = hmac.new(self._shared_key, str(element).encode(), hashlib.sha256).hexdigest()
            encrypted_map[h] = element
        return encrypted_map

    @abstractmethod
    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """计算交集"""
        pass


class OTPSIProtocol(PSIBase):
    """
    基于不经意传输(OT)的PSI协议

    使用OT协议确保双方都不知道对方的非交集元素。
    """

    def __init__(self, role: str, channel: Any = None, security_parameter: int = 128):
        """
        初始化OT PSI协议

        Args:
            role: 角色
            channel: 通信通道
            security_parameter: 安全参数
        """
        super().__init__(role, channel)
        self.security_parameter = security_parameter

    def _generate_ot_parameters(self) -> Tuple[bytes, bytes]:
        """生成OT参数"""
        k0 = secrets.token_bytes(self.security_parameter // 8)
        k1 = secrets.token_bytes(self.security_parameter // 8)
        return k0, k1

    @abstractmethod
    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """计算交集"""
        pass


class HEPSIProtocol(PSIBase):
    """
    基于全同态加密(HE)的PSI协议

    使用同态加密在密文上进行交集计算，提供最强的隐私保护。
    """

    def __init__(self, role: str, channel: Any = None, key_size: int = 2048):
        """
        初始化HE PSI协议

        Args:
            role: 角色
            channel: 通信通道
            key_size: 密钥大小
        """
        super().__init__(role, channel)
        self.key_size = key_size
        self._public_key: Optional[Tuple[int, int]] = None
        self._private_key: Optional[Tuple[int, int]] = None

    def _generate_paillier_keypair(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """生成Paillier密钥对"""
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

        bits = min(self.key_size // 2, 512)
        p = generate_prime(bits)
        q = generate_prime(bits)

        n = p * q
        g = n + 1
        lambda_n = (p - 1) * (q - 1)
        mu = pow(lambda_n, -1, n)

        return (n, g), (lambda_n, mu)

    @abstractmethod
    def compute_intersection(self, local_data: Union[List, Set, np.ndarray]) -> Set:
        """计算交集"""
        pass
