"""
Federated PSI (Private Set Intersection) Module
联邦求交模块

提供安全的多方集合交集计算功能，支持批量和实时两种模式。

支持的协议：
- DH (Diffie-Hellman): 基于密钥交换的求交协议
- OT (Oblivious Transfer): 基于不经意传输的求交协议
- HE (Homomorphic Encryption): 基于全同态加密的求交协议

支持的模式：
- Batch: 批量求交模式，适用于大规模数据集
- Realtime: 实时求交模式，适用于低延迟场景（仅DH协议支持）

算法列表：
- BatchDHPSI: 基于密钥交换DH算法的批量联邦求交
- BatchOTPSI: 基于不经意传输OT算法的批量联邦求交
- BatchHEPSI: 基于全同态加密HE算法的批量联邦求交
- RealtimeDHPSI: 基于密钥交换DH算法的实时联邦求交
"""

# 基础类（无外部依赖）
from .base import (
    PSIBase,
    DHPSIProtocol,
    OTPSIProtocol,
    HEPSIProtocol,
)

# 批量求交算法
from .batch_psi import (
    BatchDHPSI,
    BatchOTPSI,
    BatchHEPSI,
)

# 实时求交算法
from .realtime_psi import (
    RealtimeDHPSI,
)

# Host/Guest端实现（依赖运行时环境）
# 使用延迟导入以支持独立使用算法类
try:
    from .host import FederatedPSIHost
    from .guest import FederatedPSIGuest
    _host_guest_available = True
except ImportError:
    FederatedPSIHost = None
    FederatedPSIGuest = None
    _host_guest_available = False

__all__ = [
    # Base classes
    "PSIBase",
    "DHPSIProtocol",
    "OTPSIProtocol",
    "HEPSIProtocol",
    # Batch PSI
    "BatchDHPSI",
    "BatchOTPSI",
    "BatchHEPSI",
    # Realtime PSI
    "RealtimeDHPSI",
    # Host/Guest
    "FederatedPSIHost",
    "FederatedPSIGuest",
]
