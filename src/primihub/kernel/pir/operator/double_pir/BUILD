load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Toggle for linking against @simplepir//:simplepir_c_kernels (the
# upstream ahenzinger/simplepir C matrix-multiplication kernels). When
# off (default), double_pir_runtime.cc compiles into a clean stub that
# returns retcode::FAIL with a "not vendored" message, and the
# @simplepir dep is not pulled in. Turn on via:
#   bazel build --define=enable_double_pir_real=1 \
#               --override_repository=simplepir=<path>
config_setting(
    name = "enable_real",
    define_values = {"enable_double_pir_real": "1"},
)

# Phase-4 scaffolding library for DoublePIR (USENIX'23).
# Mirrors the spiral_pir_operator pattern: registers the algorithm + caps
# at static-init time via the AddPending → EnsureRegistered two-phase
# path (commit 5d86cafc). With --define=enable_double_pir_real=1 the
# OnExecute now drives DoublePirRuntime::SmokeMatMulVec which calls
# upstream matMulVec on a 4x4 known matrix, validating that the
# @simplepir cc_library link works end-to-end. Without the flag, every
# call returns retcode::FAIL with a clear "not vendored" message.
#
# kIsSkeleton stays true: even with the runtime smoke passing, the full
# DoublePIR query path (LWE secret + hint + answer + decode) is not yet
# implemented — that is the multi-day task 5.5 follow-up.
#
# alwayslink = True is mandatory — without it the linker drops the TU
# because the binary never names DoublePirOperator directly.
cc_library(
    name = "double_pir_operator",
    srcs = [
        "double_pir.cc",
        "double_pir_runtime.cc",
    ],
    hdrs = [
        "double_pir.h",
        "double_pir_runtime.h",
    ],
    defines = select({
        ":enable_real": ["PIR_DOUBLE_PIR_RUNTIME_VENDORED=1"],
        "//conditions:default": [],
    }),
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:multi_peer_pir",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_protocol",
        "//src/primihub/kernel/pir/operator/double_pir:hint_gen",
        "//src/primihub/kernel/pir/operator/double_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_github_glog_glog//:glog",
    ] + select({
        ":enable_real": ["@simplepir//:simplepir_c_kernels"],
        "//conditions:default": [],
    }),
)

# Algorithm-level static functions (Init / Setup / Query / Answer /
# Recover) for DoublePIR. Built separately from double_pir_operator
# so the operator keeps its current scaffolding flow while the
# protocol port lands in chunks 3..6 of openspec task 5.5. Setup,
# Query, Answer, Recover need Matrix kernel calls so the cc_library
# deps include pir_core::matrix which has its own enable_pir_core_real
# activation. Query also pulls pir_core::gaussian for the LWE noise.
cc_library(
    name = "double_pir_protocol",
    srcs = ["double_pir_protocol.cc"],
    hdrs = ["double_pir_protocol.h"],
    deps = [
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:gaussian",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_github_glog_glog//:glog",
    ],
)

# HintGen — refactor of the Init + Setup sequence out of the operator
# into a reusable helper. First chunk of task 5.6 (two-server hint
# precomputation). Caching + LinkContext split land as follow-up
# chunks once the data flow is settled.
cc_library(
    name = "hint_gen",
    srcs = ["hint_gen.cc"],
    hdrs = ["hint_gen.h"],
    deps = [
        ":double_pir_protocol",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_github_glog_glog//:glog",
    ],
)

# HintCache — process-local LRU cache wrapping HintGen::Compute, keyed
# by (l, m, p, logq, db fingerprint). Task 5.6 chunk 2.
cc_library(
    name = "hint_cache",
    srcs = ["hint_cache.cc"],
    hdrs = ["hint_cache.h"],
    deps = [
        ":hint_gen",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "@com_github_glog_glog//:glog",
    ],
)
