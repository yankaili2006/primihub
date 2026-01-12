
### 1、proxy error
#### error message
ERROR: no such package '@rules_foreign_cc//foreign_cc': java.io.IOException: Error downloading [https://github.com/bazelbuild/rules_foreign_cc/archive/0.5.1.tar.gz] to /private/var/tmp/_bazel_yankaili/0debc33b2ff838305d7336badf927d09/external/rules_foreign_cc/temp18356444105672555342/0.5.1.tar.gz: Proxy address 127.0.0.1:7890 is not a valid URL

#### solution
set proxy

    export http{,s}_proxy=http://127.0.0.1:7890

### 2、Mac M1 compile error
#### darwin_x86_64

  https://githubhot.com/repo/bazelbuild/tulsi/issues/272

  M1

  https://engineering.mercari.com/en/blog/entry/20211210-bazel-remote-execution-for-ios-builds-with-apple-silicon/

  https://github.com/bazelbuild/bazel/commit/e4e06c0293bda20ad8c2b8db131ce821316b8d12#diff-1f4964536f397e8679625954c7139bd6

  https://github.com/bazelbuild/bazel/commit/b235517a1f3b91561a513ea4381365dba30dd592#diff-1f4964536f397e8679625954c7139bd6

  openssl

  https://groups.google.com/g/bazel-discuss/c/NCvZ3HYyTl0?pli=1

  https://github.com/bazelbuild/bazel/issues/7032

  https://github.com/bazelbuild/rules_foreign_cc/issues/337

  https://github.com/bazelbuild/rules_foreign_cc/issues/337#issuecomment-582500380

  https://github.com/openssl/openssl/issues/3840

  https://github.com/3rdparty/bazel-rules-openssl/blob/main/BUILD.openssl.bazel

### 3、install bazel@5.0.0

  wget https://mirrors.huaweicloud.com/bazel/5.0.0/bazel-5.0.0-installer-darwin-x86_64.sh
  chmod +x ./bazel-5.0.0-installer-darwin-x86_64.sh
  ./bazel-5.0.0-installer-darwin-x86_64.sh


### 4、bazel 5.0.0

    "_scm_call_1", referenced from:
      _internal_guile_eval in guile.o
      _guile_expand_wrapper in guile.o
      _guile_eval_wrapper in guile.o
    "_scm_eval_string", referenced from:
        _internal_guile_eval in guile.o
    "_scm_from_locale_string", referenced from:
        _guile_expand_wrapper in guile.o
    "_scm_from_utf8_string", referenced from:
        _internal_guile_eval in guile.o
    "_scm_to_locale_string", referenced from:
        _internal_guile_eval in guile.o
        _guile_expand_wrapper in guile.o
        _guile_eval_wrapper in guile.o
    "_scm_variable_ref", referenced from:
        _guile_init in guile.o
    "_scm_with_guile", referenced from:
        _func_guile in guile.o
  ld: symbol(s) not found for architecture x86_64
  clang: error: linker command failed with exit code 1 (use -v to see invocation)
  _____ END BUILD LOGS _____
  rules_foreign_cc: Build wrapper script location: bazel-out/darwin-opt-exec-2B5CBBC6/bin/external/rules_foreign_cc/toolchains/make_tool_foreign_cc/wrapper_build_script.sh
  rules_foreign_cc: Build script location: bazel-out/darwin-opt-exec-2B5CBBC6/bin/external/rules_foreign_cc/toolchains/make_tool_foreign_cc/build_script.sh
  rules_foreign_cc: Build log location: bazel-out/darwin-opt-exec-2B5CBBC6/bin/external/rules_foreign_cc/toolchains/make_tool_foreign_cc/BootstrapGNUMake.log

  Target //:node failed to build
  Use --verbose_failures to see the command lines of failed build steps.

### 5. ar: only one of -a and -[bi] options allowed

    ar -static -s apps/libapps.a apps/app_rand.o apps/apps.o apps/bf_prefix.o apps/opt.o apps/s_cb.o apps/s_socket.o
    ar: only one of -a and -[bi] options allowed
    usage:  ar -d [-TLsv] archive file ...
        ar -m [-TLsv] archive file ...
        ar -m [-abiTLsv] position archive file ...
    6.0.0-pre.20220411.2
        ar -p [-TLsv] archive [file ...]
        ar -q [-cTLsv] archive file ...
        ar -r [-cuTLsv] archive file ...
        ar -r [-abciuTLsv] position archive file ...
        ar -t [-TLsv] archive [file ...]
        ar -x [-ouTLsv] archive [file ...]
    make[1]: *** [Makefile:676: apps/libapps.a] Error 1
    make[1]: Leaving directory '/private/var/tmp/_bazel_liyankai/cb6f6703cfdeb7b1b37d6df6c4eeb127/sandbox/darwin-sandbox/399/execroot/__main__/bazel-out/darwin_arm64-fastbuild/bin/external/openssl/openssl.build_tmpdir'
    make: *** [Makefile:177: build_libs] Error 2
    make: Leaving directory '/private/var/tmp/_bazel_liyankai/cb6f6703cfdeb7b1b37d6df6c4eeb127/sandbox/darwin-sandbox/399/execroot/__main__/bazel-out/darwin_arm64-fastbuild/bin/external/openssl/openssl.build_tmpdir'

### 6. Microsoft APSI 编译错误 - digit_t 类型未定义

#### 错误信息
```
external/mircrosoft_apsi/common/apsi/fourq/eccp2_core.c:504:49: error: 'digit_t' undeclared
external/mircrosoft_apsi/common/apsi/fourq/eccp2_core.c:504:57: error: expected ')' before 'scalar'
```

#### 错误原因
Microsoft APSI 依赖库中的 FourQ 密码库存在架构兼容性问题，`digit_t` 类型在特定编译环境下未正确定义。

#### 解决方案
**临时解决方案（开发测试用）：**
1. 创建简单的 task_main 替代程序：
```bash
g++ -std=c++17 -o bazel-bin/task_main simple_task_main.cc
```

**完整解决方案：**
1. 修复 Microsoft APSI 依赖库的架构兼容性问题
2. 确保 FourQ 库正确配置 digit_t 类型定义
3. 重新构建完整的 task_main

### 7. C++17 兼容性错误

#### 错误信息
```
external/ph_communication/network/channel_interface.h:405:49: error: expected unqualified-id before ',' token
external/ph_communication/network/channel_interface.h:524:23: error: 'string_view' is not a member of 'std'
```

#### 错误原因
外部依赖库 `ph_communication` 使用了 C++17 特性（如 `std::string_view`），但在某些编译环境下未正确配置 C++17 标准。

#### 解决方案
1. 确保编译环境支持 C++17
2. 在 .bazelrc 中正确配置 C++17 编译选项
3. 更新外部依赖库以兼容 C++17 标准

### 8. task_main 缺失导致任务执行失败

#### 错误信息
```
ERROR: 72
task execute encountes error
```

#### 错误原因
PrimiHub 系统默认使用 PROCESS 模式执行任务，需要调用 `task_main` 可执行文件。当 `task_main` 不存在时，任务执行失败并返回错误码 72。

#### 解决方案
**方案一：使用 THREAD 模式（推荐用于开发）**
修改 `src/primihub/node/worker/worker.h`：
```cpp
TaskRunMode task_run_mode_{TaskRunMode::THREAD};
// TaskRunMode task_run_mode_{TaskRunMode::PROCESS};
```

**方案二：提供 task_main 替代程序**
创建简单的 task_main 可执行文件作为临时解决方案。

### 9. 文件系统命名空间错误

#### 错误信息
```
src/primihub/common/config/config.cc:34:3: error: 'fs' has not been declared
```

#### 错误原因
C++17 文件系统库的命名空间使用问题。

#### 解决方案
修改 `src/primihub/common/config/config.cc`：
```cpp
#include <filesystem>
namespace fs = std::filesystem;
```

