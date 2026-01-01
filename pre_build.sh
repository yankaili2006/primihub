#!/bin/bash
set -x
set -e

echo "================================================"
echo "PrimiHub 构建前配置脚本"
echo "版本: 2.0 (包含问题解决方案)"
echo "================================================"

# 检查并设置代理（解决网络下载问题）
check_and_set_proxy() {
    echo "检查网络代理设置..."
    
    # 如果环境变量中已有代理，使用现有设置
    if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
        echo "使用环境变量中的代理: $http_proxy"
        return 0
    fi
    
    # 检查Clash代理
    if pgrep -x "clash" > /dev/null; then
        echo "检测到Clash代理，设置代理为: http://127.0.0.1:7890"
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export HTTPS_PROXY=http://127.0.0.1:7890
        return 0
    fi
    
    echo "提示: 未设置代理，网络下载可能较慢"
    echo "如需设置代理，请执行:"
    echo "  export http_proxy=http://127.0.0.1:7890"
    echo "  export https_proxy=http://127.0.0.1:7890"
    return 1
}

# 检查系统依赖（解决编译依赖问题）
check_system_dependencies() {
    echo "检查系统依赖..."
    
    local missing_deps=""
    
    # 检查m4（解决GMP构建失败问题）
    if ! command -v m4 >/dev/null 2>&1; then
        missing_deps="$missing_deps m4"
        echo "错误: 缺少m4工具，会导致GMP库构建失败"
        echo "解决方案: sudo apt-get install m4"
    fi
    
    # 检查MySQL开发库（解决MySQL驱动编译问题）
    if ! pkg-config --exists libmysqlclient 2>/dev/null; then
        if [ ! -f /usr/include/mysql/mysql.h ] && [ ! -f /usr/local/include/mysql/mysql.h ]; then
            missing_deps="$missing_deps libmysqlclient-dev"
            echo "警告: 缺少MySQL开发库，编译时将跳过MySQL支持"
            echo "解决方案: sudo apt-get install libmysqlclient-dev"
        fi
    fi
    
    if [ -n "$missing_deps" ]; then
        echo "================================================"
        echo "重要: 缺少以下依赖，可能影响构建:"
        echo "$missing_deps"
        echo "================================================"
        echo "按回车继续（构建可能失败），或Ctrl+C中断安装依赖"
        read -r
    fi
}

# 执行代理检查
check_and_set_proxy

# 执行依赖检查
check_system_dependencies

PYTHON_BIN=python3
if ! command -v python3 >/dev/null 2>&1; then
  if ! command -v python >/dev/null 2>&1; then
    echo "please install python3"
    exit
  else
    PYTHON_BIN=python
  fi
fi
U_V1=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
U_V2=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
U_V3=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $3}'`

echo your python version is : "$U_V1.$U_V2.$U_V3"
if ! [ "${U_V1}" = 3 ] && [ "${U_V2}" > 6 ]; then
  echo "python version must > 3.6"
  exit
fi

PYTHON_CONFIG_CMD="python$U_V1.$U_V2-config"

if ! command -v ${PYTHON_CONFIG_CMD} >/dev/null 2>&1; then
  echo "please install python$U_V1.$U_V2-dev"
  exit
fi

#get python include path
PYTHON_INC_CONFIG=`${PYTHON_CONFIG_CMD} --includes | awk '{print $1}' |awk -F'-I' '{print $2}'`
if [ ! -d "${PYTHON_INC_CONFIG}" ]; then
  echo "${PYTHON_CONFIG_CMD} get python include path failed"
  exit -1
fi

# link python include path into workspace
pushd third_party
rm -f python_headers
ln -s ${PYTHON_INC_CONFIG} python_headers
popd

#get python link option
CONFIG=`${PYTHON_CONFIG_CMD} --ldflags` && NEWLINE="[\"${CONFIG}\"] + [\"-lpython$U_V1.$U_V2\"]"

# Compatible with MacOS
sed -e "s|PLACEHOLDER-PYTHON3.X-CONFIG|${NEWLINE}|g" BUILD.bazel > BUILD.bazel.tmp && mv BUILD.bazel.tmp BUILD.bazel
echo "done"

if [ ! -d "localdb" ]; then
    mkdir localdb
fi

if [ ! -d "log" ]; then
    mkdir log
fi

#detect platform and machine hardware
KERNEL_NAME=$(uname -s)
KERNEL_NAME=$(echo $KERNEL_NAME | tr '[:upper:]' '[:lower:]')
MACHINE_HARDWARE=$(uname -m)
sed -e "s|PLATFORM_HARDWARE|${KERNEL_NAME}_${MACHINE_HARDWARE}|g" Makefile > Makefile.tmp && mv Makefile.tmp Makefile

echo ""
echo "================================================"
echo "构建前配置完成！"
echo "================================================"
echo ""
echo "常见构建命令:"
echo "1. 标准构建: make release"
echo "2. 带MySQL支持: make release mysql=y"
echo "3. 跳过MySQL支持: make release mysql="
echo "4. 多线程构建 (4线程): make release jobs=4"
echo ""
echo "如果构建失败，请检查:"
echo "1. 网络代理设置是否正确"
echo "2. 系统依赖是否安装完整"
echo "3. 查看错误日志中的具体信息"
echo "================================================"
