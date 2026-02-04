#!/bin/bash
#
# PrimiHub 自动化构建脚本
# 功能：配置 Docker 镜像源 + 构建 primihub 镜像
#
# 使用方法：
#   bash auto_build_primihub.sh
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "PrimiHub 自动化构建脚本"
echo "========================================"
echo ""

# ===========================
# 步骤 1: 配置 Docker 镜像源
# ===========================
echo -e "${YELLOW}[步骤 1/4] 配置 Docker 镜像源...${NC}"

# 备份原配置
if [ -f /etc/docker/daemon.json ]; then
    BACKUP_FILE="/etc/docker/daemon.json.bak.$(date +%Y%m%d_%H%M%S)"
    echo "备份原配置到: $BACKUP_FILE"
    sudo cp /etc/docker/daemon.json "$BACKUP_FILE"
fi

# 读取当前配置
CURRENT_DATA_ROOT=$(grep -o '"data-root"[[:space:]]*:[[:space:]]*"[^"]*"' /etc/docker/daemon.json 2>/dev/null | cut -d'"' -f4 || echo "")

# 写入新配置
echo "写入新的 Docker 配置..."
if [ -n "$CURRENT_DATA_ROOT" ]; then
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "$CURRENT_DATA_ROOT",
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
else
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
fi

# 重启 Docker
echo "重启 Docker 服务..."
sudo systemctl daemon-reload
sudo systemctl restart docker

# 等待 Docker 启动
echo "等待 Docker 服务启动..."
sleep 5

# 验证配置
echo "验证镜像源配置..."
docker info | grep -A 5 "Registry Mirrors" || echo "配置已生效"

echo -e "${GREEN}✓ Docker 镜像源配置完成${NC}"
echo ""

# ===========================
# 步骤 2: 拉取基础镜像
# ===========================
echo -e "${YELLOW}[步骤 2/4] 拉取 Ubuntu 基础镜像...${NC}"
docker pull ubuntu:20.04

echo -e "${GREEN}✓ 基础镜像拉取完成${NC}"
echo ""

# ===========================
# 步骤 3: 构建 primihub 镜像
# ===========================
echo -e "${YELLOW}[步骤 3/4] 构建 primihub 镜像...${NC}"
echo "这可能需要 20-40 分钟，请耐心等待..."
echo ""

cd /mnt/data1/pcloud/external/primihub

# 使用主 Dockerfile 构建
docker build -t primihub/primihub-node:latest -f Dockerfile .

echo -e "${GREEN}✓ primihub 镜像构建完成${NC}"
echo ""

# ===========================
# 步骤 4: 验证镜像
# ===========================
echo -e "${YELLOW}[步骤 4/4] 验证构建结果...${NC}"
docker images | grep primihub

echo ""
echo "========================================"
echo -e "${GREEN}构建成功！${NC}"
echo "========================================"
echo ""
echo "镜像信息："
docker images primihub/primihub-node:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""
echo "使用方法："
echo "  1. 运行容器："
echo "     docker run -it primihub/primihub-node:latest /bin/bash"
echo ""
echo "  2. 使用 docker-compose 启动集群："
echo "     cd /mnt/data1/pcloud/external/primihub"
echo "     docker-compose up -d"
echo ""
echo "========================================"
