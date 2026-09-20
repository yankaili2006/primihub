load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Toggle for linking against the real menonsamir/spiral upstream library.
# Off by default — when off, spiral_runtime.cc compiles into a clean stub
# that returns retcode::FAIL with a "not vendored" message, and no @hexl /
# @spiral_pir deps are required. Turn on via:
#   bazel build --define=enable_spiral_real=1 \
#               --override_repository=hexl=<path>      \
#               --override_repository=spiral_pir=<path>
config_setting(
    name = "enable_real",
    define_values = {"enable_spiral_real": "1"},
)

# Operator library. Registers SpiralPirOperator + capabilities so the
# Selector can see it. OnExecute is currently a clear-error stub; the
# real cryptographic path lives behind --define=enable_spiral_real=1
# in spiral_runtime.cc. See openspec/changes/primihub-pir-multi-algo
# tasks 4.4 / 4.5 for the implementation plan.
#
# alwayslink = True is mandatory: the registrar instance is the only thing
# in spiral_pir.cc that the binary references, so without alwayslink the
# linker would strip the whole TU and the algorithm would silently not
# register.
cc_library(
    name = "spiral_pir_operator",
    srcs = [
        "spiral_pir.cc",
        "spiral_runtime.cc",
    ],
    hdrs = [
        "params.h",
        "spiral_pir.h",
        "spiral_runtime.h",
    ],
    defines = select({
        ":enable_real": ["PIR_SPIRAL_RUNTIME_VENDORED=1"],
        "//conditions:default": [],
    }),
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ] + select({
        ":enable_real": [
            "@spiral_pir//:spiral_lib",
            "@hexl//:hexl",
        ],
        "//conditions:default": [],
    }),
)
