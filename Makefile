# ================================================
# PrimiHub Makefile
# 版本: 2.0 (增强版)
# 包含常见问题解决方案和优化选项
# ================================================

BUILD_FLAG ?=

# 默认构建目标
TARGET := //:node \
          //:cli \
          //:task_main

# ================================================
# 构建选项
# ================================================

# 功能开关
ifneq ($(disable_py_task), y)
  TARGET += //src/primihub/pybind_warpper:linkcontext \
      //src/primihub/pybind_warpper:opt_paillier_c2py \
      //src/primihub/task/pybind_wrapper:ph_secure_lib
  BUILD_FLAG += --define enable_py_task=true
endif

ifneq ($(disable_mpc_task), y)
  BUILD_FLAG += --define enable_mpc_task=true
endif

ifneq ($(disable_pir_task), y)
  BUILD_FLAG += --define enable_pir_task=true
endif

ifneq ($(disable_psi_task), y)
  BUILD_FLAG += --define enable_psi_task=true
endif

# MySQL支持 (需要安装libmysqlclient-dev)
ifeq ($(mysql), y)
  BUILD_FLAG += --define enable_mysql_driver=true
  $(info [INFO] 启用MySQL驱动支持)
else
  $(info [INFO] 禁用MySQL驱动支持 (如需启用请使用 mysql=y))
endif

# 协议生成
ifeq ($(protos), y)
  TARGET += //src/primihub/protos:worker_py_pb2_grpc \
      //src/primihub/protos:service_py_pb2_grpc
endif

# 并行构建 (加速构建过程)
JOBS?=
ifneq ($(jobs), )
	JOBS = $(jobs)
	BUILD_FLAG += --jobs=$(JOBS)
	$(info [INFO] 使用 $(JOBS) 个线程并行构建)
endif

# TEE/SGX支持
ifeq ($(tee), y)
	BUILD_FLAG += --cxxopt=-DSGX
	BUILD_FLAG += --define enable_sgx=true
	$(info [INFO] 启用TEE/SGX支持)
endif

# 调试构建 (包含asan检测)
ifeq ($(debug), y)
	BUILD_FLAG += --config=linux_asan
	$(info [INFO] 启用调试模式 (ASAN))
endif

# 详细输出 (显示详细构建信息)
ifeq ($(verbose), y)
	BUILD_FLAG += --subcommands --verbose_failures
	$(info [INFO] 启用详细输出模式)
endif

# ================================================
# 构建规则
# ================================================

.PHONY: help
help:
	@echo "================================================"
	@echo "PrimiHub 构建系统"
	@echo "================================================"
	@echo ""
	@echo "常用构建命令:"
	@echo "  make release              # 标准构建"
	@echo "  make release mysql=y      # 启用MySQL支持"
	@echo "  make release jobs=4       # 4线程并行构建"
	@echo "  make release debug=y      # 调试构建"
	@echo "  make release tee=y        # 启用TEE/SGX"
	@echo "  make release verbose=y    # 详细输出"
	@echo ""
	@echo "组合选项:"
	@echo "  make release mysql=y jobs=4"
	@echo ""
	@echo "其他命令:"
	@echo "  make clean                # 清理构建缓存"
	@echo "  make help                 # 显示此帮助"
	@echo ""
	@echo "常见问题:"
	@echo "  1. 缺少m4工具: sudo apt-get install m4"
	@echo "  2. 缺少MySQL开发库: sudo apt-get install libmysqlclient-dev"
	@echo "  3. 网络问题: 设置代理 export http_proxy=http://127.0.0.1:7890"
	@echo "================================================"

.PHONY: release
release:
	@echo "================================================"
	@echo "开始构建 PrimiHub..."
	@echo "构建选项: $(BUILD_FLAG)"
	@echo "构建目标: $(TARGET)"
	@echo "================================================"
	@bazel build --config=linux_x86_64 $(BUILD_FLAG) ${TARGET}
	@rm -f primihub-cli
	@ln -s -f bazel-bin/cli primihub-cli
	@rm -f primihub-node
	@ln -s -f bazel-bin/node primihub-node
	@echo "================================================"
	@echo "构建完成！"
	@echo "生成的可执行文件:"
	@echo "  primihub-node -> bazel-bin/node"
	@echo "  primihub-cli  -> bazel-bin/cli"
	@echo "================================================"
	@echo "下一步:"
	@echo "  1. 启动服务: ./start_server.sh"
	@echo "  2. 运行测试: ./primihub-cli --task_config_file=\"example/psi_ecdh_task_conf.json\""
	@echo "  3. 停止服务: ./stop_server.sh"
	@echo "================================================"

# 平台特定构建 (已注释，按需启用)
#linux_x86_64:
#	bazel build --config=linux_x86_64 ${TARGET}
#
#linux_aarch64:
#	bazel build --config=linux_aarch64 ${TARGET}
#
#macos_arm64:
#	bazel build --config=darwin_arm64 ${TARGET}
#
#macos_x86_64:
#	bazel build --config=darwin_x86_64 ${TARGET}

.PHONY: clean
clean:
	@echo "清理构建缓存..."
	@bazel clean
	@echo "清理完成"

# ================================================
# 快速测试命令
# ================================================

.PHONY: test-psi
test-psi:
	@echo "测试PSI功能..."
	@./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"

.PHONY: test-mpc
test-mpc:
	@echo "测试MPC功能..."
	@./primihub-cli --task_config_file="example/mpc_lr_task_conf.json"

.PHONY: test-all
test-all: test-psi test-mpc
	@echo "所有测试完成"

# ================================================
# 开发辅助命令
# ================================================

.PHONY: deps
deps:
	@echo "安装Python依赖..."
	@cd python && pip install -r requirements.txt

.PHONY: format
format:
	@echo "格式化代码 (需要安装clang-format)..."
	@find src -name "*.cc" -o -name "*.h" | xargs clang-format -i

.PHONY: lint
lint:
	@echo "代码检查 (需要安装cpplint)..."
	@find src -name "*.cc" -o -name "*.h" | xargs python3 external/cpplint.py --filter=-build/include_subdir

# ================================================
# 默认目标
# ================================================
.DEFAULT_GOAL := help
