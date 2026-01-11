#!/bin/bash
#
# Docker镜像构建脚本
# 用法:
#   bash build_docker.sh [MODE] [TAG] [IMAGE_NAME]
#
# 参数:
#   MODE        - 编译模式: FULL (完整版) 或 MINI (精简版)，默认: FULL
#   TAG         - 镜像标签，默认: 当前时间戳
#   IMAGE_NAME  - 镜像名称，默认: primihub/primihub-node
#
# 示例:
#   bash build_docker.sh FULL 2024-01-08 192.168.99.10/primihub/primihub-node
#   bash build_docker.sh MINI latest primihub/primihub-node
#

set -e

# ===========================
# 参数解析
# ===========================
MODE=${1:-FULL}
TAG=${2:-$(date +%F-%H-%M-%S)}
IMAGE_NAME=${3:-primihub/primihub-node}

# ===========================
# 配置变量
# ===========================
BUILD_TIMESTAMP=$(date +%s)
COMMIT_FILE="commit.txt"

# ===========================
# 打印构建信息
# ===========================
echo "========================================"
echo "PrimiHub Docker 镜像构建"
echo "========================================"
echo "编译模式:     $MODE"
echo "镜像标签:     $TAG"
echo "镜像名称:     $IMAGE_NAME"
echo "构建时间戳:   $BUILD_TIMESTAMP"
echo "========================================"

# ===========================
# 预处理: 根据模式调整配置
# ===========================
if [ "$MODE" = "FULL" ]; then
    echo "[FULL模式] 准备完整版构建..."

    # 删除 requirements.txt 第一行（如果存在特殊配置）
    if [ -f "python/requirements.txt" ]; then
        echo "移除 python/requirements.txt 第一行"
        sed -i '1d' python/requirements.txt
    fi

    FINAL_TAG="$TAG"

elif [ "$MODE" = "MINI" ]; then
    echo "[MINI模式] 准备精简版构建..."

    FINAL_TAG="mini-$TAG"

else
    echo "错误: 未知的编译模式 '$MODE'"
    echo "支持的模式: FULL, MINI"
    exit 1
fi

# ===========================
# 步骤 1: 运行预构建脚本
# ===========================
echo ""
echo "[步骤 1/5] 运行预构建脚本..."
if [ -f "pre_build.sh" ]; then
    bash pre_build.sh
else
    echo "警告: pre_build.sh 不存在，跳过预构建"
fi

# ===========================
# 步骤 2: 编译项目
# ===========================
echo ""
echo "[步骤 2/5] 使用 Bazel 编译项目..."
make release mysql=y

if [ $? -ne 0 ]; then
    echo "错误: 编译失败!"
    exit 1
fi

# ===========================
# 步骤 3: 记录版本信息
# ===========================
echo ""
echo "[步骤 3/5] 记录版本信息..."
git rev-parse --abbrev-ref HEAD > $COMMIT_FILE 2>/dev/null || echo "unknown-branch" > $COMMIT_FILE
git rev-parse HEAD >> $COMMIT_FILE 2>/dev/null || echo "unknown-commit" >> $COMMIT_FILE
echo "$BUILD_TIMESTAMP" >> $COMMIT_FILE

cat $COMMIT_FILE

# ===========================
# 步骤 4: 打包构建产物
# ===========================
echo ""
echo "[步骤 4/5] 打包构建产物..."
tar zcf bazel-bin.tar.gz \
    bazel-bin/cli \
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
    $COMMIT_FILE

if [ $? -ne 0 ]; then
    echo "错误: 打包失败!"
    exit 1
fi

# ===========================
# 步骤 5: 构建 Docker 镜像
# ===========================
echo ""
echo "[步骤 5/5] 构建 Docker 镜像..."
docker build -t ${IMAGE_NAME}:${FINAL_TAG} -f Dockerfile.local .

if [ $? -ne 0 ]; then
    echo "错误: Docker 镜像构建失败!"
    exit 1
fi

# ===========================
# 清理临时文件
# ===========================
echo ""
echo "清理临时文件..."
rm -f $COMMIT_FILE

# ===========================
# 构建完成
# ===========================
echo ""
echo "========================================"
echo "构建成功!"
echo "========================================"
echo "镜像: ${IMAGE_NAME}:${FINAL_TAG}"
echo "时间戳: $BUILD_TIMESTAMP"
echo "========================================"
echo ""
echo "推送镜像到仓库:"
echo "  docker push ${IMAGE_NAME}:${FINAL_TAG}"
echo ""
echo "运行容器:"
echo "  docker run -it ${IMAGE_NAME}:${FINAL_TAG} /bin/bash"
echo "========================================"
