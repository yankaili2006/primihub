"""
联邦求交使用示例
Federated PSI Examples

演示如何使用各种联邦求交算法。
"""

import sys
sys.path.insert(0, '/home/primihub/github/primihub/python')

from primihub.FL.federated_psi import (
    BatchDHPSI, BatchOTPSI, BatchHEPSI,
    RealtimeDHPSI
)


class MockChannel:
    """模拟通信通道用于测试"""

    def __init__(self, peer_channel=None):
        self._data = {}
        self._peer = peer_channel

    def set_peer(self, peer_channel):
        self._peer = peer_channel

    def send(self, key: str, value):
        if self._peer:
            self._peer._data[key] = value

    def recv(self, key: str):
        import time
        max_wait = 5
        start = time.time()
        while key not in self._data:
            if time.time() - start > max_wait:
                raise TimeoutError(f"Timeout waiting for key: {key}")
            time.sleep(0.01)
        return self._data.pop(key)


def example_batch_dh_psi():
    """
    示例：基于DH算法的批量联邦求交

    场景：
    - 两家公司想要找出共同的客户
    - 双方都不希望泄露各自的完整客户列表
    """
    print("=" * 60)
    print("示例：基于DH算法的批量联邦求交")
    print("=" * 60)

    # 公司A的客户ID列表
    company_a_customers = {
        "customer_001", "customer_002", "customer_003",
        "customer_004", "customer_005"
    }

    # 公司B的客户ID列表
    company_b_customers = {
        "customer_003", "customer_004", "customer_006",
        "customer_007", "customer_008"
    }

    print(f"公司A客户数: {len(company_a_customers)}")
    print(f"公司B客户数: {len(company_b_customers)}")
    expected_intersection = company_a_customers & company_b_customers
    print(f"预期交集: {expected_intersection}")
    print()

    # 创建通信通道
    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    # 创建PSI实例
    host_psi = BatchDHPSI(role="host", channel=host_channel)
    guest_psi = BatchDHPSI(role="guest", channel=guest_channel)

    # 并行执行
    import threading

    host_result = set()
    guest_result = set()

    def host_task():
        nonlocal host_result
        host_result = host_psi.compute_intersection(company_a_customers)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_psi.compute_intersection(company_b_customers)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"Host方求交结果: {host_result}")
    print(f"Guest方求交结果: {guest_result}")
    print(f"结果验证: {'通过' if host_result == expected_intersection else '失败'}")
    print()


def example_batch_ot_psi():
    """
    示例：基于OT算法的批量联邦求交

    OT协议提供更强的隐私保证，
    双方都无法获知对方的非交集元素。
    """
    print("=" * 60)
    print("示例：基于OT算法的批量联邦求交")
    print("=" * 60)

    set_a = {"apple", "banana", "cherry", "date"}
    set_b = {"banana", "cherry", "elderberry", "fig"}

    print(f"集合A: {set_a}")
    print(f"集合B: {set_b}")
    expected = set_a & set_b
    print(f"预期交集: {expected}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    host_psi = BatchOTPSI(role="host", channel=host_channel)
    guest_psi = BatchOTPSI(role="guest", channel=guest_channel)

    import threading

    host_result = set()
    guest_result = set()

    def host_task():
        nonlocal host_result
        host_result = host_psi.compute_intersection(set_a)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_psi.compute_intersection(set_b)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"Host方结果: {host_result}")
    print(f"Guest方结果: {guest_result}")
    print(f"结果验证: {'通过' if host_result == expected else '失败'}")
    print()


def example_batch_he_psi():
    """
    示例：基于HE算法的批量联邦求交

    同态加密提供最高级别的安全保证，
    计算在密文上进行，明文从不暴露。
    """
    print("=" * 60)
    print("示例：基于HE算法的批量联邦求交")
    print("=" * 60)

    set_a = {1001, 1002, 1003, 1004, 1005}
    set_b = {1003, 1004, 1005, 1006, 1007}

    print(f"集合A: {set_a}")
    print(f"集合B: {set_b}")
    expected = set_a & set_b
    print(f"预期交集: {expected}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    host_psi = BatchHEPSI(role="host", channel=host_channel)
    guest_psi = BatchHEPSI(role="guest", channel=guest_channel)

    import threading

    host_result = set()
    guest_result = set()

    def host_task():
        nonlocal host_result
        host_result = host_psi.compute_intersection(set_a)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_psi.compute_intersection(set_b)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"Host方结果: {host_result}")
    print(f"Guest方结果: {guest_result}")
    print(f"结果验证: {'通过' if host_result == expected else '失败'}")
    print()


def example_realtime_dh_psi():
    """
    示例：基于DH算法的实时联邦求交

    实时PSI特点：
    - 会话密钥复用
    - 支持增量求交
    - 低延迟响应
    """
    print("=" * 60)
    print("示例：基于DH算法的实时联邦求交")
    print("=" * 60)

    set_a = {"user_1", "user_2", "user_3", "user_4"}
    set_b = {"user_2", "user_3", "user_5", "user_6"}

    print(f"集合A: {set_a}")
    print(f"集合B: {set_b}")
    expected = set_a & set_b
    print(f"预期交集: {expected}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    host_psi = RealtimeDHPSI(role="host", channel=host_channel)
    guest_psi = RealtimeDHPSI(role="guest", channel=guest_channel)

    import threading

    host_result = set()
    guest_result = set()

    def host_task():
        nonlocal host_result
        host_result = host_psi.compute_intersection(set_a)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_psi.compute_intersection(set_b)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"Host方结果: {host_result}")
    print(f"Guest方结果: {guest_result}")
    print(f"结果验证: {'通过' if host_result == expected else '失败'}")
    print()


def example_psi_with_large_dataset():
    """
    示例：大规模数据集的联邦求交

    演示批量处理大量数据的能力。
    """
    print("=" * 60)
    print("示例：大规模数据集的联邦求交")
    print("=" * 60)

    import random

    # 生成大规模数据集
    n_host = 10000
    n_guest = 8000
    overlap = 2000

    # 创建有重叠的数据集
    common_ids = {f"id_{i}" for i in range(overlap)}
    host_only = {f"host_{i}" for i in range(n_host - overlap)}
    guest_only = {f"guest_{i}" for i in range(n_guest - overlap)}

    set_a = common_ids | host_only
    set_b = common_ids | guest_only

    print(f"集合A大小: {len(set_a)}")
    print(f"集合B大小: {len(set_b)}")
    print(f"预期交集大小: {len(common_ids)}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    # 使用较大的batch_size以提高效率
    host_psi = BatchDHPSI(role="host", channel=host_channel, batch_size=5000)
    guest_psi = BatchDHPSI(role="guest", channel=guest_channel, batch_size=5000)

    import threading
    import time

    host_result = set()
    guest_result = set()

    start_time = time.time()

    def host_task():
        nonlocal host_result
        host_result = host_psi.compute_intersection(set_a)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_psi.compute_intersection(set_b)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    elapsed_time = time.time() - start_time

    print(f"求交完成，耗时: {elapsed_time:.2f} 秒")
    print(f"实际交集大小: {len(host_result)}")
    print(f"结果验证: {'通过' if host_result == common_ids else '失败'}")
    print()


def main():
    """运行所有示例"""
    print("\n联邦求交算法使用示例\n")

    example_batch_dh_psi()
    example_batch_ot_psi()
    example_batch_he_psi()
    example_realtime_dh_psi()
    example_psi_with_large_dataset()

    print("=" * 60)
    print("所有示例执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
