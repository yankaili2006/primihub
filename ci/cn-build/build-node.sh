#!/bin/bash
set -e
export GIT_SSH_COMMAND="ssh -F /dev/null -i /root/.ssh_rw/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh_rw/known_hosts -o BatchMode=yes"
export no_proxy=aliyuncs.com,localhost,127.0.0.1 NO_PROXY=aliyuncs.com,localhost,127.0.0.1
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
export USE_BAZEL_VERSION=5.0.0 CC=/usr/bin/gcc-8 CXX=/usr/bin/g++-8
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
git config --global http.lowSpeedLimit 0
cd /src
# node binary embeds linkcontext (PYBIND11_EMBEDDED_MODULE) via :node; also build the .so + others
bazelisk build --config=linux_x86_64 --define enable_py_task=true --distdir=/distdir \
  :node :cli //:task_main \
  //src/primihub/pybind_wrapper:linkcontext \
  //src/primihub/pybind_wrapper:opt_paillier_c2py \
  //src/primihub/task/pybind_wrapper:ph_secure_lib
cp -f bazel-bin/src/primihub/pybind_wrapper/linkcontext.so python/ || true
cp -f bazel-bin/src/primihub/pybind_wrapper/opt_paillier_c2py.so python/ || true
cp -f bazel-bin/src/primihub/task/pybind_wrapper/ph_secure_lib.so python/ || true
cp -f bazel-bin/src/primihub/protos/*.py python/primihub/client/ph_grpc/src/primihub/protos/ 2>/dev/null || true
ln -sf bazel-bin/node primihub-node
ln -sf bazel-bin/cli primihub-cli
tar zcf primihub-linux-amd64.tar.gz bazel-bin/cli bazel-bin/node primihub-cli primihub-node \
  bazel-bin/task_main \
  bazel-bin/src/primihub/pybind_wrapper/opt_paillier_c2py.so \
  bazel-bin/src/primihub/pybind_wrapper/linkcontext.so \
  bazel-bin/src/primihub/task/pybind_wrapper/ph_secure_lib.so \
  start_server.sh stop_server.sh client_run.sh python config example data
echo "BUILD OK; tarball bytes: $(stat -c%s primihub-linux-amd64.tar.gz)"
