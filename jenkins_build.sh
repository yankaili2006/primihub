#!/bin/bash
#
# Jenkins CI/CD Docker镜像构建和推送脚本
# 用法:
#   bash jenkins_build.sh [COMPILE_MODE] [REGISTRY]
#
# 参数:
#   COMPILE_MODE  - 编译模式: ALL (完整版) 或 MINI (精简版)，默认: ALL
#   REGISTRY      - Docker仓库地址，默认: 192.168.99.10
#
# 环境变量:
#   BUILD_TIMESTAMP - 构建时间戳（可选，默认自动生成）
#   PUSH_IMAGE      - 是否推送镜像 (yes/no)，默认: yes
#   ALIYUN_PUSH     - 是否推送到阿里云 (yes/no)，默认: no
#
# 示例:
#   bash jenkins_build.sh ALL 192.168.99.10
#   PUSH_IMAGE=no bash jenkins_build.sh MINI
#   ALIYUN_PUSH=yes bash jenkins_build.sh ALL
#

set -e

# ===========================
# 颜色输出
# ===========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ===========================
# 参数解析
# ===========================
COMPILE_MODE=${1:-ALL}
REGISTRY=${2:-192.168.99.10}
BUILD_TIMESTAMP=${BUILD_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}
PUSH_IMAGE=${PUSH_IMAGE:-yes}
ALIYUN_PUSH=${ALIYUN_PUSH:-no}

# 镜像配置
IMAGE_NAME="${REGISTRY}/primihub/primihub-node"
ALIYUN_REGISTRY="registry.cn-beijing.aliyuncs.com/primihub/primihub-node"

# ===========================
# 打印构建信息
# ===========================
echo ""
log_info "========================================"
log_info "PrimiHub CI/CD 镜像构建"
log_info "========================================"
log_info "编译模式:       ${COMPILE_MODE}"
log_info "镜像仓库:       ${REGISTRY}"
log_info "镜像名称:       ${IMAGE_NAME}"
log_info "构建时间戳:     ${BUILD_TIMESTAMP}"
log_info "推送镜像:       ${PUSH_IMAGE}"
log_info "推送阿里云:     ${ALIYUN_PUSH}"
log_info "========================================"
echo ""

# ===========================
# 步骤 1: 预处理
# ===========================
log_info "步骤 1: 预处理配置..."

if [ "$COMPILE_MODE" = "ALL" ]; then
    log_info "[ALL模式] 准备完整版构建"

    # 删除 requirements.txt 第一行
    if [ -f "python/requirements.txt" ]; then
        log_info "移除 python/requirements.txt 第一行"
        sed -i '1d' python/requirements.txt
    fi

    BUILD_MODE="FULL"
    IMAGE_TAG="${BUILD_TIMESTAMP}"

elif [ "$COMPILE_MODE" = "MINI" ]; then
    log_info "[MINI模式] 准备精简版构建"

    BUILD_MODE="MINI"
    IMAGE_TAG="mini-${BUILD_TIMESTAMP}"

else
    log_error "未知的编译模式: ${COMPILE_MODE}"
    log_error "支持的模式: ALL, MINI"
    exit 1
fi

# ===========================
# 步骤 2: 运行构建脚本
# ===========================
log_info "步骤 2: 执行构建..."

if [ -f "build_local.sh" ]; then
    bash build_local.sh ${BUILD_MODE} ${BUILD_TIMESTAMP} ${IMAGE_NAME}

    if [ $? -ne 0 ]; then
        log_error "构建失败!"
        exit 1
    fi
else
    log_error "build_local.sh 不存在!"
    exit 1
fi

log_success "构建完成: ${IMAGE_NAME}:${IMAGE_TAG}"

# ===========================
# 步骤 3: 推送镜像到私有仓库
# ===========================
if [ "$PUSH_IMAGE" = "yes" ]; then
    log_info "步骤 3: 推送镜像到 ${REGISTRY}..."

    docker push ${IMAGE_NAME}:${IMAGE_TAG}

    if [ $? -eq 0 ]; then
        log_success "推送成功: ${IMAGE_NAME}:${IMAGE_TAG}"
    else
        log_error "推送失败: ${IMAGE_NAME}:${IMAGE_TAG}"
        exit 1
    fi
else
    log_warn "跳过镜像推送 (PUSH_IMAGE=${PUSH_IMAGE})"
fi

# ===========================
# 步骤 4: 推送到阿里云镜像仓库
# ===========================
if [ "$ALIYUN_PUSH" = "yes" ]; then
    log_info "步骤 4: 推送镜像到阿里云..."

    # Tag 镜像
    log_info "标记镜像: ${ALIYUN_REGISTRY}:${BUILD_TIMESTAMP}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ALIYUN_REGISTRY}:${BUILD_TIMESTAMP}

    if [ $? -ne 0 ]; then
        log_error "镜像标记失败!"
        exit 1
    fi

    # Push 镜像
    log_info "推送镜像到阿里云..."
    docker push ${ALIYUN_REGISTRY}:${BUILD_TIMESTAMP}

    if [ $? -eq 0 ]; then
        log_success "阿里云推送成功: ${ALIYUN_REGISTRY}:${BUILD_TIMESTAMP}"
    else
        log_error "阿里云推送失败!"
        exit 1
    fi
else
    log_info "跳过阿里云推送 (ALIYUN_PUSH=${ALIYUN_PUSH})"
fi

# ===========================
# 步骤 5: 清理
# ===========================
log_info "步骤 5: 清理临时文件..."

if [ -f "commit.txt" ]; then
    rm -f commit.txt
    log_info "已删除 commit.txt"
fi

# ===========================
# 构建完成
# ===========================
echo ""
log_success "========================================"
log_success "构建流程完成!"
log_success "========================================"
echo ""
log_info "构建信息:"
log_info "  编译模式:   ${COMPILE_MODE}"
log_info "  镜像标签:   ${IMAGE_TAG}"
log_info "  时间戳:     ${BUILD_TIMESTAMP}"
echo ""
log_info "镜像列表:"
log_info "  私有仓库:   ${IMAGE_NAME}:${IMAGE_TAG}"
if [ "$ALIYUN_PUSH" = "yes" ]; then
    log_info "  阿里云:     ${ALIYUN_REGISTRY}:${BUILD_TIMESTAMP}"
fi
echo ""
log_info "使用镜像:"
log_info "  docker pull ${IMAGE_NAME}:${IMAGE_TAG}"
log_info "  docker run -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash"
echo ""
log_success "========================================"
