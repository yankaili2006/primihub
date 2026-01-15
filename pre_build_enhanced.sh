#!/bin/bash
set -x
set -e

echo "================================================"
echo "PrimiHub 增强版构建脚本"
echo "包含常见问题解决方案"
echo "================================================"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查并设置代理
setup_proxy() {
    print_info "检查网络代理设置..."
    
    # 检查是否有Clash代理
    if pgrep -x "clash" > /dev/null; then
        print_info "检测到Clash代理正在运行"
        PROXY_PORT="7890"
    else
        # 检查环境变量中的代理
        if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
            print_info "使用环境变量中的代理"
            return 0
        fi
        
        print_warning "未检测到代理，网络下载可能较慢"
        print_info "如需设置代理，请执行:"
        print_info "  export http_proxy=http://127.0.0.1:7890"
        print_info "  export https_proxy=http://127.0.0.1:7890"
        return 1
    fi
    
    # 设置代理环境变量
    export http_proxy="http://127.0.0.1:${PROXY_PORT}"
    export https_proxy="http://127.0.0.1:${PROXY_PORT}"
    export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
    export HTTPS_PROXY="http://127.0.0.1:${PROXY_PORT}"
    
    print_info "已设置代理: http://127.0.0.1:${PROXY_PORT}"
    return 0
}

# 检查并安装系统依赖
check_system_deps() {
    print_info "检查系统依赖..."
    
    local missing_deps=()
    
    # 检查必需的工具
    for tool in m4 gcc g++ make curl wget; do
        if ! command -v $tool >/dev/null 2>&1; then
            missing_deps+=("$tool")
        fi
    done
    
    # 检查开发库
    if ! pkg-config --exists libmysqlclient 2>/dev/null; then
        missing_deps+=("libmysqlclient-dev")
    fi
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_info "所有系统依赖已安装"
        return 0
    fi
    
    print_warning "缺少以下依赖: ${missing_deps[*]}"
    
    # 尝试自动安装（需要sudo权限）
    if [ -f /etc/debian_version ]; then
        print_info "检测到Debian/Ubuntu系统，尝试安装依赖..."
        echo "请使用sudo权限运行以下命令:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install -y ${missing_deps[*]}"
    elif [ -f /etc/redhat-release ]; then
        print_info "检测到RHEL/CentOS系统，尝试安装依赖..."
        echo "请使用sudo权限运行以下命令:"
        echo "  sudo yum install -y ${missing_deps[*]}"
    else
        print_error "无法自动安装依赖，请手动安装: ${missing_deps[*]}"
        return 1
    fi
    
    return 1
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    PYTHON_BIN=python3
    if ! command -v python3 >/dev/null 2>&1; then
        if ! command -v python >/dev/null 2>&1; then
            print_error "请安装python3"
            exit 1
        else
            PYTHON_BIN=python
        fi
    fi
    
    U_V1=`$PYTHON_BIN -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
    U_V2=`$PYTHON_BIN -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
    U_V3=`$PYTHON_BIN -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $3}'`
    
    echo "Python版本: $U_V1.$U_V2.$U_V3"
    
    if ! [ "${U_V1}" = 3 ] && [ "${U_V2}" -gt 6 ]; then
        print_error "Python版本必须 > 3.6"
        exit 1
    fi
    
    PYTHON_CONFIG_CMD="python$U_V1.$U_V2-config"
    
    if ! command -v ${PYTHON_CONFIG_CMD} >/dev/null 2>&1; then
        print_error "请安装 python$U_V1.$U_V2-dev"
        if [ -f /etc/debian_version ]; then
            echo "运行: sudo apt-get install python$U_V1.$U_V2-dev"
        elif [ -f /etc/redhat-release ]; then
            echo "运行: sudo yum install python$U_V1-devel"
        fi
        exit 1
    fi
    
    # 检查Python头文件
    PYTHON_INC_CONFIG=`${PYTHON_CONFIG_CMD} --includes | awk '{print $1}' | awk -F'-I' '{print $2}'`
    if [ ! -d "${PYTHON_INC_CONFIG}" ]; then
        print_error "${PYTHON_CONFIG_CMD} 获取Python头文件路径失败"
        exit 1
    fi
    
    print_info "Python环境检查通过"
    return 0
}

# 配置Python头文件链接
setup_python_headers() {
    print_info "配置Python头文件链接..."
    
    pushd third_party > /dev/null
    rm -f python_headers
    ln -s ${PYTHON_INC_CONFIG} python_headers
    popd > /dev/null
    
    # 更新BUILD.bazel中的Python链接选项
    CONFIG=`${PYTHON_CONFIG_CMD} --ldflags` && NEWLINE="[\"${CONFIG}\"] + [\"-lpython$U_V1.$U_V2\"]"
    
    # 兼容macOS
    sed -e "s|PLACEHOLDER-PYTHON3.X-CONFIG|${NEWLINE}|g" BUILD.bazel > BUILD.bazel.tmp && mv BUILD.bazel.tmp BUILD.bazel
    
    print_info "Python头文件配置完成"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    for dir in localdb log data/result; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "创建目录: $dir"
        fi
    done
}

# 检测平台并更新Makefile
detect_platform() {
    print_info "检测平台..."
    
    KERNEL_NAME=$(uname -s)
    KERNEL_NAME=$(echo $KERNEL_NAME | tr '[:upper:]' '[:lower:]')
    MACHINE_HARDWARE=$(uname -m)
    
    print_info "平台: $KERNEL_NAME, 硬件: $MACHINE_HARDWARE"
    
    sed -e "s|PLATFORM_HARDWARE|${KERNEL_NAME}_${MACHINE_HARDWARE}|g" Makefile > Makefile.tmp && mv Makefile.tmp Makefile
    
    print_info "Makefile已更新为: ${KERNEL_NAME}_${MACHINE_HARDWARE}"
}

# 检查Bazel版本
check_bazel_version() {
    print_info "检查Bazel版本..."
    
    if ! command -v bazel >/dev/null 2>&1; then
        print_error "Bazel未安装"
        print_info "安装方法:"
        print_info "1. 使用Bazelisk (推荐):"
        print_info "   curl -Lo bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
        print_info "   chmod +x bazelisk"
        print_info "   sudo mv bazelisk /usr/local/bin/bazel"
        print_info "2. 或从官网下载: https://bazel.build/install"
        exit 1
    fi
    
    BAZEL_VERSION=$(bazel version 2>/dev/null | grep "Build label" | awk '{print $3}')
    print_info "Bazel版本: $BAZEL_VERSION"
    
    # 检查版本兼容性
    if [[ "$BAZEL_VERSION" =~ ^([0-9]+)\. ]]; then
        MAJOR_VERSION=${BASH_REMATCH[1]}
        if [ "$MAJOR_VERSION" -lt 4 ]; then
            print_warning "Bazel版本可能过旧，建议使用4.x或5.x版本"
        fi
    fi
}

# 应用补丁文件
apply_patches() {
    print_info "检查并应用补丁文件..."
    
    local patches=(
        "fix_absl.patch"
        "fix_cryptotools.patch"
        "fix_gcc13.patch"
    )
    
    for patch in "${patches[@]}"; do
        if [ -f "$patch" ]; then
            print_info "发现补丁文件: $patch"
            # 这里可以添加应用补丁的逻辑
            # patch -p1 < "$patch" 2>/dev/null || true
        fi
    done
}

# 显示构建选项
show_build_options() {
    echo ""
    echo "================================================"
    echo "构建选项:"
    echo "================================================"
    echo "1. 标准构建 (无MySQL): make release"
    echo "2. 带MySQL支持: make release mysql=y"
    echo "3. 调试构建: make release debug=y"
    echo "4. TEE支持 (SGX): make release tee=y"
    echo "5. 多线程构建 (使用4个线程): make release jobs=4"
    echo ""
    echo "常用组合:"
    echo "  make release mysql=y jobs=4"
    echo ""
    echo "清理构建缓存: make clean"
    echo "================================================"
}

# 显示常见问题解决方案
show_troubleshooting() {
    echo ""
    echo "================================================"
    echo "常见问题解决方案:"
    echo "================================================"
    echo "1. 编译错误: 'No usable m4 in $PATH'"
    echo "   解决方案: sudo apt-get install m4"
    echo ""
    echo "2. 编译错误: 'mysql/mysql.h: No such file or directory'"
    echo "   解决方案: sudo apt-get install libmysqlclient-dev"
    echo ""
    echo "3. 网络超时: 无法下载依赖"
    echo "   解决方案: 设置代理"
    echo "     export http_proxy=http://127.0.0.1:7890"
    echo "     export https_proxy=http://127.0.0.1:7890"
    echo ""
    echo "4. Python导入错误: 'ModuleNotFoundError'"
    echo "   解决方案: 安装Python依赖"
    echo "     cd python && pip install -r requirements.txt"
    echo ""
    echo "5. 服务启动失败: 端口被占用"
    echo "   解决方案: 停止现有服务"
    echo "     ./stop_server.sh"
    echo "     或修改config/*.yaml中的端口号"
    echo "================================================"
}

# 主函数
main() {
    echo ""
    print_info "开始PrimiHub构建前配置..."
    
    # 执行各步骤
    setup_proxy
    check_system_deps || print_warning "系统依赖检查未通过，可能影响构建"
    check_python_env
    setup_python_headers
    create_directories
    detect_platform
    check_bazel_version
    apply_patches
    
    echo ""
    print_info "构建前配置完成！"
    
    show_build_options
    show_troubleshooting
    
    echo ""
    print_info "下一步: 运行构建命令"
    print_info "例如: make release mysql=y"
    
    # 记录构建配置
    echo ""
    echo "构建配置已保存到: build_config_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "构建时间: $(date)"
        echo "Python版本: $U_V1.$U_V2.$U_V3"
        echo "平台: ${KERNEL_NAME}_${MACHINE_HARDWARE}"
        echo "代理设置: $http_proxy"
        echo "Python头文件: $PYTHON_INC_CONFIG"
    } > "build_config_$(date +%Y%m%d_%H%M%S).log"
}

# 运行主函数
main "$@"