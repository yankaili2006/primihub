#!/bin/bash
# Docker 镜像源配置脚本

set -e

echo "================================================"
echo "配置 Docker 国内镜像源"
echo "================================================"

# 备份原配置
echo "1. 备份原配置文件..."
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%Y%m%d_%H%M%S)

# 写入新配置
echo "2. 配置镜像源..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "/mnt/disks/sdf/docker",
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com"
  ]
}
EOF

# 重启 Docker
echo "3. 重启 Docker 服务..."
sudo systemctl daemon-reload
sudo systemctl restart docker

# 等待 Docker 启动
echo "4. 等待 Docker 服务启动..."
sleep 3

# 验证配置
echo "5. 验证配置..."
docker info | grep -A 5 "Registry Mirrors" || echo "配置已生效"

echo ""
echo "================================================"
echo "配置完成！"
echo "================================================"
echo "现在可以正常拉取镜像了"
echo ""
