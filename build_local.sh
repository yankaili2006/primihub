#!/bin/bash
#
# 本地Docker镜像构建脚本
# When using this script, please make sure that the python version of the local machine is 3.8
#
# 用法:
#   bash build_local.sh [MODE] [TAG] [IMAGE_NAME]
#
# 参数:
#   MODE        - 编译模式: FULL/MINI (可选，默认 FULL)
#   TAG         - 镜像标签 (默认: 当前时间戳)
#   IMAGE_NAME  - 镜像名称 (默认: primihub/primihub-node)
#

set -x
mode_full="FULL"

# 参数解析
[[ -z "$1" ]] && PRIMIHUB_MODE=$mode_full
[[ -n "$1" ]] && PRIMIHUB_MODE=$1

if [[ "$PRIMIHUB_MODE" != "${mode_full}" ]]; then
  PRIMIHUB_MODE="MINI"
fi

if [ ! -n "$2" ] ; then
    TAG=`date +%F-%H-%M-%S`
else
    TAG=$2
fi

if [ ! -n "$3" ] ; then
    IMAGE_NAME="primihub/primihub-node"
else
    IMAGE_NAME=$3
fi

# 根据模式调整镜像标签
if [ "$PRIMIHUB_MODE" = "MINI" ]; then
    FINAL_TAG="mini-${TAG}"
else
    FINAL_TAG="${TAG}"
fi

echo "========================================"
echo "构建模式: $PRIMIHUB_MODE"
echo "镜像标签: $FINAL_TAG"
echo "镜像名称: $IMAGE_NAME"
echo "========================================"

bash pre_build.sh "$PRIMIHUB_MODE"

build_opt="mysql=y"
if [[ "$PRIMIHUB_MODE" == "FULL" ]]; then
  build_opt="${build_opt}"
else
  build_opt="${build_opt} \
    disable_py_task=y  \
  "
  # build_opt="${build_opt} \
  #   disable_py_task=y  \
  #   disable_mpc_task=y \
  #   disable_pir_task=y \
  #   disable_psi_task=y \
  # "
fi

make release $build_opt

if [ $? -ne 0 ]; then
    echo "Build failed!!!"
    exit
fi

git rev-parse --abbrev-ref HEAD >> commit.txt
git rev-parse HEAD >> commit.txt
release_pkg="bazel-bin/cli \
  bazel-bin/node \
  bazel-bin/_solib* \
  bazel-bin/task_main \
  config \
  example \
  data \
  commit.txt \
"

if [[ "$PRIMIHUB_MODE" == "FULL" ]]; then
release_pkg="${release_pkg} \
  bazel-bin/src/primihub/pybind_warpper/opt_paillier_c2py.so \
  bazel-bin/src/primihub/pybind_warpper/linkcontext.so \
  bazel-bin/src/primihub/task/pybind_wrapper/ph_secure_lib.so \
  python \
"
fi

tar zcfh bazel-bin.tar.gz ${release_pkg}

if [[ "$PRIMIHUB_MODE" == "FULL" ]]; then
  docker_file=Dockerfile.local
  docker build -t $IMAGE_NAME:$FINAL_TAG . -f ${docker_file}
else
  docker_file=Dockerfile.mini
  docker build -t $IMAGE_NAME:$FINAL_TAG . -f ${docker_file}
fi

echo ""
echo "========================================"
echo "构建完成!"
echo "镜像: $IMAGE_NAME:$FINAL_TAG"
echo "========================================"
