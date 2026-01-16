"""
Federated Query Module
联邦查询模块

提供安全的多方数据查询功能，支持批量和实时两种模式。

支持的协议：
- DH (Diffie-Hellman): 基于密钥交换的查询协议
- OT (Oblivious Transfer): 基于不经意传输的查询协议
- HE (Homomorphic Encryption): 基于全同态加密的查询协议

支持的模式：
- Batch: 批量查询模式，适用于大规模数据查询
- Realtime: 实时查询模式，适用于低延迟场景

算法列表：
- BatchDHQuery: 基于密钥交换DH算法的批量联邦查询
- BatchOTQuery: 基于不经意传输OT算法的批量联邦查询
- BatchHEQuery: 基于全同态加密HE算法的批量联邦查询
- RealtimeDHQuery: 基于密钥交换DH算法的实时联邦查询
- RealtimeOTQuery: 基于不经意传输OT算法的实时联邦查询
- RealtimeHEQuery: 基于全同态加密HE算法的实时联邦查询
"""

# 基础类（无外部依赖）
from .base import (
    FederatedQueryBase,
    DHQueryProtocol,
    OTQueryProtocol,
    HEQueryProtocol,
)

# 批量查询算法
from .batch_query import (
    BatchDHQuery,
    BatchOTQuery,
    BatchHEQuery,
)

# 实时查询算法
from .realtime_query import (
    RealtimeDHQuery,
    RealtimeOTQuery,
    RealtimeHEQuery,
)

# Host/Guest端实现（依赖运行时环境）
# 使用延迟导入以支持独立使用算法类
try:
    from .host import FederatedQueryHost
    from .guest import FederatedQueryGuest
    _host_guest_available = True
except ImportError:
    FederatedQueryHost = None
    FederatedQueryGuest = None
    _host_guest_available = False

__all__ = [
    # Base classes
    "FederatedQueryBase",
    "DHQueryProtocol",
    "OTQueryProtocol",
    "HEQueryProtocol",
    # Batch query
    "BatchDHQuery",
    "BatchOTQuery",
    "BatchHEQuery",
    # Realtime query
    "RealtimeDHQuery",
    "RealtimeOTQuery",
    "RealtimeHEQuery",
    # Host/Guest
    "FederatedQueryHost",
    "FederatedQueryGuest",
]
