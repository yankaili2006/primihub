"""
联邦查询使用示例
Federated Query Examples

演示如何使用各种联邦查询算法。
"""

import sys
sys.path.insert(0, '/home/primihub/github/primihub/python')

from primihub.FL.federated_query import (
    BatchDHQuery, BatchOTQuery, BatchHEQuery,
    RealtimeDHQuery, RealtimeOTQuery, RealtimeHEQuery
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


def example_batch_dh_query():
    """
    示例：基于DH算法的批量联邦查询

    场景：
    - Host方有用户ID列表，想要查询这些用户在Guest方的数据
    - 双方都不希望泄露各自的完整数据集
    """
    print("=" * 60)
    print("示例：基于DH算法的批量联邦查询")
    print("=" * 60)

    # 模拟Host方数据（查询方）
    host_data = {
        "user_001": {"name": "Alice", "age": 25},
        "user_002": {"name": "Bob", "age": 30},
        "user_003": {"name": "Charlie", "age": 35},
        "user_004": {"name": "David", "age": 40},
    }
    host_keys = set(host_data.keys())

    # 模拟Guest方数据（数据方）
    guest_data = {
        "user_002": {"score": 85, "level": "A"},
        "user_003": {"score": 92, "level": "A+"},
        "user_005": {"score": 78, "level": "B"},
        "user_006": {"score": 88, "level": "A"},
    }
    guest_keys = set(guest_data.keys())

    print(f"Host方有 {len(host_keys)} 条记录: {host_keys}")
    print(f"Guest方有 {len(guest_keys)} 条记录: {guest_keys}")
    print()

    # 创建通信通道
    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    # 创建查询实例
    host_query = BatchDHQuery(role="host", channel=host_channel)
    guest_query = BatchDHQuery(role="guest", channel=guest_channel)

    # 并行执行（实际部署中在不同节点）
    import threading

    host_result = {}
    guest_result = {}

    def host_task():
        nonlocal host_result
        host_result = host_query.execute_query(host_keys, host_data)

    def guest_task():
        nonlocal guest_result
        guest_result = guest_query.execute_query(guest_keys, guest_data)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"查询结果（Host方可见的匹配记录）: {len(host_result)} 条")
    for key, value in host_result.items():
        print(f"  {key}: {value}")
    print()


def example_batch_ot_query():
    """
    示例：基于OT算法的批量联邦查询

    OT（不经意传输）提供更强的隐私保护：
    - Guest方不知道Host方具体查询了哪些数据
    - Host方只能获取查询结果，无法获取其他数据
    """
    print("=" * 60)
    print("示例：基于OT算法的批量联邦查询")
    print("=" * 60)

    host_keys = {"id_A", "id_B", "id_C"}
    guest_data = {
        "id_A": "value_A",
        "id_B": "value_B",
        "id_D": "value_D",
    }

    print(f"Host方查询: {host_keys}")
    print(f"Guest方数据: {set(guest_data.keys())}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    host_query = BatchOTQuery(role="host", channel=host_channel)
    guest_query = BatchOTQuery(role="guest", channel=guest_channel)

    import threading

    host_result = {}

    def host_task():
        nonlocal host_result
        host_result = host_query.execute_query(host_keys)

    def guest_task():
        guest_query.execute_query(set(guest_data.keys()), guest_data)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"Host方获取到 {len(host_result)} 条匹配记录")
    for key, value in host_result.items():
        print(f"  {key}: {value}")
    print()


def example_realtime_dh_query():
    """
    示例：基于DH算法的实时联邦查询

    实时查询特点：
    - 低延迟响应
    - 支持会话复用
    - 适合在线场景
    """
    print("=" * 60)
    print("示例：基于DH算法的实时联邦查询")
    print("=" * 60)

    host_keys = {"key_1", "key_2", "key_3"}
    guest_data = {
        "key_1": "data_1",
        "key_2": "data_2",
        "key_4": "data_4",
    }

    print(f"Host方查询: {host_keys}")
    print(f"Guest方数据: {set(guest_data.keys())}")
    print()

    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)

    host_query = RealtimeDHQuery(role="host", channel=host_channel)
    guest_query = RealtimeDHQuery(role="guest", channel=guest_channel)

    import threading

    host_result = {}

    def host_task():
        nonlocal host_result
        host_result = host_query.execute_query(host_keys)

    def guest_task():
        guest_query.execute_query(set(guest_data.keys()), guest_data)

    host_thread = threading.Thread(target=host_task)
    guest_thread = threading.Thread(target=guest_task)

    host_thread.start()
    guest_thread.start()

    host_thread.join()
    guest_thread.join()

    print(f"实时查询结果: {len(host_result)} 条匹配")
    for key, value in host_result.items():
        print(f"  {key}: {value}")
    print()


def main():
    """运行所有示例"""
    print("\n联邦查询算法使用示例\n")

    example_batch_dh_query()
    example_batch_ot_query()
    example_realtime_dh_query()

    print("=" * 60)
    print("所有示例执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
