#!/bin/bash
#
# 本地Docker镜像构建脚本
# When using this script, please make sure that the python version of the local machine is 3.8
#
# 用法:
#   bash build_local.sh [MODE] [TAG] [IMAGE_NAME]
#   bash build_local.sh [TAG] [IMAGE_NAME]  # 兼容旧版本
#
# 参数:
#   MODE        - 编译模式: FULL/MINI (可选，默认 FULL)
#   TAG         - 镜像标签 (默认: 当前时间戳)
#   IMAGE_NAME  - 镜像名称 (默认: primihub/primihub-node)
#

# 参数解析：支持新旧两种格式
if [ -n "$1" ] && { [ "$1" = "FULL" ] || [ "$1" = "MINI" ]; }; then
    # 新格式: MODE TAG IMAGE_NAME
    MODE=$1
    TAG=${2:-`date +%F-%H-%M-%S`}
    IMAGE_NAME=${3:-"primihub/primihub-node"}
else
    # 旧格式: TAG IMAGE_NAME (兼容)
    MODE="FULL"
    TAG=${1:-`date +%F-%H-%M-%S`}
    IMAGE_NAME=${2:-"primihub/primihub-node"}
fi

# 根据模式调整镜像标签
if [ "$MODE" = "MINI" ]; then
    FINAL_TAG="mini-${TAG}"
else
    FINAL_TAG="${TAG}"
fi

echo "========================================"
echo "构建模式: $MODE"
echo "镜像标签: $FINAL_TAG"
echo "镜像名称: $IMAGE_NAME"
echo "========================================"

bash pre_build.sh

make release mysql=y

if [ $? -ne 0 ]; then
    echo "Build failed!!!"
    exit
fi

git rev-parse --abbrev-ref HEAD >> commit.txt
git rev-parse HEAD >> commit.txt

tar zcf bazel-bin.tar.gz bazel-bin/cli \
        bazel-bin/node \
        primihub-cli \
        primihub-node \
        bazel-bin/task_main \
        bazel-bin/src/primihub/pybind_warpper/opt_paillier_c2py.so \
        bazel-bin/src/primihub/pybind_warpper/linkcontext.so \
        bazel-bin/src/primihub/task/pybind_wrapper/ph_secure_lib.so \
        python \
        config \
        example \
        data \
        commit.txt

docker build -t $IMAGE_NAME:$FINAL_TAG . -f Dockerfile.local

echo ""
echo "========================================"
echo "构建完成!"
echo "镜像: $IMAGE_NAME:$FINAL_TAG"
echo "========================================"