"""
联邦查询和联邦求交单元测试
Unit Tests for Federated Query and PSI
"""

import sys
import unittest
import threading
from typing import Any, Dict, Set

sys.path.insert(0, '/home/primihub/github/primihub/python')


class MockChannel:
    """模拟通信通道"""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._peer: 'MockChannel' = None

    def set_peer(self, peer: 'MockChannel'):
        self._peer = peer

    def send(self, key: str, value: Any):
        if self._peer:
            self._peer._data[key] = value

    def recv(self, key: str) -> Any:
        import time
        max_wait = 10
        start = time.time()
        while key not in self._data:
            if time.time() - start > max_wait:
                raise TimeoutError(f"Timeout waiting for key: {key}")
            time.sleep(0.01)
        return self._data.pop(key)


def create_channel_pair():
    """创建一对连接的通道"""
    host_channel = MockChannel()
    guest_channel = MockChannel()
    host_channel.set_peer(guest_channel)
    guest_channel.set_peer(host_channel)
    return host_channel, guest_channel


class TestFederatedQuery(unittest.TestCase):
    """联邦查询单元测试"""

    def test_batch_dh_query(self):
        """测试BatchDHQuery"""
        from primihub.FL.federated_query import BatchDHQuery

        host_keys = {"key1", "key2", "key3"}
        host_data = {"key1": "v1", "key2": "v2", "key3": "v3"}
        guest_keys = {"key2", "key3", "key4"}
        guest_data = {"key2": "g2", "key3": "g3", "key4": "g4"}

        host_channel, guest_channel = create_channel_pair()

        host_query = BatchDHQuery(role="host", channel=host_channel)
        guest_query = BatchDHQuery(role="guest", channel=guest_channel)

        host_result = {}
        guest_result = {}

        def host_task():
            nonlocal host_result
            host_result = host_query.execute_query(host_keys, host_data)

        def guest_task():
            nonlocal guest_result
            guest_result = guest_query.execute_query(guest_keys, guest_data)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 验证交集结果
        expected_keys = host_keys & guest_keys
        self.assertEqual(set(host_result.keys()), expected_keys)
        print(f"BatchDHQuery: 通过 - 匹配 {len(host_result)} 条记录")

    def test_batch_ot_query(self):
        """测试BatchOTQuery"""
        from primihub.FL.federated_query import BatchOTQuery

        host_keys = {"a", "b", "c"}
        guest_keys = {"b", "c", "d"}
        guest_data = {"b": "B", "c": "C", "d": "D"}

        host_channel, guest_channel = create_channel_pair()

        host_query = BatchOTQuery(role="host", channel=host_channel)
        guest_query = BatchOTQuery(role="guest", channel=guest_channel)

        host_result = {}

        def host_task():
            nonlocal host_result
            host_result = host_query.execute_query(host_keys)

        def guest_task():
            guest_query.execute_query(guest_keys, guest_data)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        expected_keys = host_keys & guest_keys
        self.assertEqual(set(host_result.keys()), expected_keys)
        print(f"BatchOTQuery: 通过 - 匹配 {len(host_result)} 条记录")

    def test_realtime_dh_query(self):
        """测试RealtimeDHQuery"""
        from primihub.FL.federated_query import RealtimeDHQuery

        host_keys = {"x", "y", "z"}
        guest_keys = {"y", "z", "w"}
        guest_data = {"y": "Y", "z": "Z", "w": "W"}

        host_channel, guest_channel = create_channel_pair()

        host_query = RealtimeDHQuery(role="host", channel=host_channel)
        guest_query = RealtimeDHQuery(role="guest", channel=guest_channel)

        host_result = {}

        def host_task():
            nonlocal host_result
            host_result = host_query.execute_query(host_keys)

        def guest_task():
            guest_query.execute_query(guest_keys, guest_data)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        expected_keys = host_keys & guest_keys
        self.assertEqual(set(host_result.keys()), expected_keys)
        print(f"RealtimeDHQuery: 通过 - 匹配 {len(host_result)} 条记录")


class TestFederatedPSI(unittest.TestCase):
    """联邦求交单元测试"""

    def test_batch_dh_psi(self):
        """测试BatchDHPSI"""
        from primihub.FL.federated_psi import BatchDHPSI

        set_a = {"1", "2", "3", "4", "5"}
        set_b = {"3", "4", "5", "6", "7"}
        expected = set_a & set_b

        host_channel, guest_channel = create_channel_pair()

        host_psi = BatchDHPSI(role="host", channel=host_channel)
        guest_psi = BatchDHPSI(role="guest", channel=guest_channel)

        host_result = set()
        guest_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            nonlocal guest_result
            guest_result = guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(host_result, expected)
        self.assertEqual(guest_result, expected)
        print(f"BatchDHPSI: 通过 - 交集大小 {len(host_result)}")

    def test_batch_ot_psi(self):
        """测试BatchOTPSI"""
        from primihub.FL.federated_psi import BatchOTPSI

        set_a = {"apple", "banana", "cherry"}
        set_b = {"banana", "cherry", "date"}
        expected = set_a & set_b

        host_channel, guest_channel = create_channel_pair()

        host_psi = BatchOTPSI(role="host", channel=host_channel)
        guest_psi = BatchOTPSI(role="guest", channel=guest_channel)

        host_result = set()
        guest_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            nonlocal guest_result
            guest_result = guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(host_result, expected)
        self.assertEqual(guest_result, expected)
        print(f"BatchOTPSI: 通过 - 交集大小 {len(host_result)}")

    def test_batch_he_psi(self):
        """测试BatchHEPSI"""
        from primihub.FL.federated_psi import BatchHEPSI

        set_a = {100, 200, 300, 400}
        set_b = {200, 300, 500, 600}
        expected = set_a & set_b

        host_channel, guest_channel = create_channel_pair()

        host_psi = BatchHEPSI(role="host", channel=host_channel)
        guest_psi = BatchHEPSI(role="guest", channel=guest_channel)

        host_result = set()
        guest_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            nonlocal guest_result
            guest_result = guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(host_result, expected)
        self.assertEqual(guest_result, expected)
        print(f"BatchHEPSI: 通过 - 交集大小 {len(host_result)}")

    def test_realtime_dh_psi(self):
        """测试RealtimeDHPSI"""
        from primihub.FL.federated_psi import RealtimeDHPSI

        set_a = {"user1", "user2", "user3"}
        set_b = {"user2", "user3", "user4"}
        expected = set_a & set_b

        host_channel, guest_channel = create_channel_pair()

        host_psi = RealtimeDHPSI(role="host", channel=host_channel)
        guest_psi = RealtimeDHPSI(role="guest", channel=guest_channel)

        host_result = set()
        guest_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            nonlocal guest_result
            guest_result = guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(host_result, expected)
        self.assertEqual(guest_result, expected)
        print(f"RealtimeDHPSI: 通过 - 交集大小 {len(host_result)}")

    def test_empty_intersection(self):
        """测试空交集情况"""
        from primihub.FL.federated_psi import BatchDHPSI

        set_a = {"a", "b", "c"}
        set_b = {"x", "y", "z"}

        host_channel, guest_channel = create_channel_pair()

        host_psi = BatchDHPSI(role="host", channel=host_channel)
        guest_psi = BatchDHPSI(role="guest", channel=guest_channel)

        host_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(host_result), 0)
        print("空交集测试: 通过")

    def test_full_overlap(self):
        """测试完全重叠情况"""
        from primihub.FL.federated_psi import BatchDHPSI

        set_a = {"1", "2", "3"}
        set_b = {"1", "2", "3"}

        host_channel, guest_channel = create_channel_pair()

        host_psi = BatchDHPSI(role="host", channel=host_channel)
        guest_psi = BatchDHPSI(role="guest", channel=guest_channel)

        host_result = set()

        def host_task():
            nonlocal host_result
            host_result = host_psi.compute_intersection(set_a)

        def guest_task():
            guest_psi.compute_intersection(set_b)

        t1 = threading.Thread(target=host_task)
        t2 = threading.Thread(target=guest_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(host_result, set_a)
        print("完全重叠测试: 通过")


class TestBaseClasses(unittest.TestCase):
    """基类测试"""

    def test_query_base_abstract(self):
        """测试FederatedQueryBase是抽象类"""
        from primihub.FL.federated_query import FederatedQueryBase

        with self.assertRaises(TypeError):
            FederatedQueryBase(role="host")
        print("FederatedQueryBase抽象类测试: 通过")

    def test_psi_base_abstract(self):
        """测试PSIBase是抽象类"""
        from primihub.FL.federated_psi import PSIBase

        with self.assertRaises(TypeError):
            PSIBase(role="host")
        print("PSIBase抽象类测试: 通过")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("联邦查询和联邦求交单元测试")
    print("=" * 60)
    print()

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFederatedQuery))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedPSI))
    suite.addTests(loader.loadTestsFromTestCase(TestBaseClasses))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("所有测试通过!")
    else:
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
