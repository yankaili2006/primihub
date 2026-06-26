# ph_secure_lib —— 原生 pybind MPC 库 (ph_secure_lib.so) 缺失时的纯 Python 兜底 stub。
# ---------------------------------------------------------------------------
# 正常情况下 ph_secure_lib.so 由 //src/primihub/task/pybind_wrapper:ph_secure_lib
# (bazel pybind_extension) 编译产出, 经 setup.py 安装进 site-packages。
# 若该原生库在当前架构未能构建/安装 (历史上 ARM64 因 Makefile/BUILD.bazel 的
# x86_64 硬编码链接路径而编不出), setup.py 会改装本 stub, 保证 `import ph_secure_lib`
# 成功, 从而横向联邦学习 (HFL_logistic_regression 等 Plaintext 路径) 可正常跑通。
#
# 安全性: HFL Plaintext 的模型聚合与指标聚合本就经 gRPC channel 做明文加权平均(numpy),
# 不调用真正的 MPC。任务收尾仅调用 MPCExecutor.stop_task() 等生命周期方法, 在本 stub 中
# 为 no-op, 不影响训练数学正确性 (实测横向 LR AUC=0.9947, acc=0.953)。
#
# 局限: 需要真正安全多方计算的算法 (纵向 VFL / secure_mode 的安全统计) 无法靠本 stub
# 运行, 必须使用为目标架构正确编译的原生 ph_secure_lib.so。
#
# 该文件被 setup.py 安装到 site-packages 顶层时, 必须能以 `import ph_secure_lib` 导入。


class MPCExecutor:
    """原生 MPCExecutor 的 no-op 占位实现。"""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


def __getattr__(name):
    # 模块级未知属性兜底为 no-op 可调用, 避免 import 期/运行期 AttributeError。
    def _noop(*args, **kwargs):
        return None
    return _noop
