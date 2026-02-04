# PrimiHub 镜像构建指南

## 问题说明

由于 Docker 无法连接到 Docker Hub，需要配置国内镜像源。但配置镜像源需要 sudo 权限。

## 解决方案

我已经为你创建了一个自动化构建脚本，你只需要执行一次并输入 sudo 密码即可完成所有操作。

## 执行步骤

### 方式 1：一键自动化构建（推荐）

```bash
cd /mnt/data1/pcloud/external/primihub
bash auto_build_primihub.sh
```

这个脚本会自动完成：
1. 配置 Docker 镜像源
2. 重启 Docker 服务
3. 拉取 Ubuntu 基础镜像
4. 构建 primihub 镜像

**预计时间：20-40 分钟**

### 方式 2：分步执行

如果你想分步执行，可以按以下步骤操作：

#### 步骤 1：配置 Docker 镜像源

```bash
cd /mnt/data1/pcloud/external/primihub
bash setup_docker_mirror.sh
```

#### 步骤 2：验证配置

```bash
docker info | grep -A 5 "Registry Mirrors"
```

#### 步骤 3：拉取基础镜像

```bash
docker pull ubuntu:20.04
```

#### 步骤 4：构建 primihub 镜像

```bash
cd /mnt/data1/pcloud/external/primihub
docker build -t primihub/primihub-node:latest -f Dockerfile .
```

## 构建完成后

### 查看镜像

```bash
docker images | grep primihub
```

### 运行单个容器

```bash
docker run -it primihub/primihub-node:latest /bin/bash
```

### 使用 docker-compose 启动集群

```bash
cd /mnt/data1/pcloud/external/primihub
docker-compose up -d
```

### 查看集群状态

```bash
docker-compose ps
```

### 查看日志

```bash
docker-compose logs -f node0
```

## 故障排除

### 如果构建失败

1. 检查网络连接
2. 确认 Docker 镜像源配置是否生效：
   ```bash
   docker info | grep -A 5 "Registry Mirrors"
   ```
3. 查看详细错误日志

### 如果需要重新构建

```bash
cd /mnt/data1/pcloud/external/primihub
docker build --no-cache -t primihub/primihub-node:latest -f Dockerfile .
```

## 相关文件

- `auto_build_primihub.sh` - 一键自动化构建脚本
- `setup_docker_mirror.sh` - Docker 镜像源配置脚本
- `Dockerfile` - 主构建文件
- `docker-compose.yml` - 集群部署配置

## 注意事项

1. 构建过程需要较长时间（20-40 分钟），请耐心等待
2. 确保有足够的磁盘空间（至少 10GB）
3. 构建过程中会下载大量依赖，请确保网络稳定
