load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# PIRANA operator (zju-abclab/PIRANA, eprint 2022/1401). SEAL+Kuku crypto
# lives in @pirana//:pirana_lib (vendored at thirdparty/pir/pirana, see
# BUILD.pirana). No build flag: deps are already in the default WORKSPACE,
# so pirana registers unconditionally via its alwayslink PirRegistrar —
# mirrors frodo_pir, not the flag-gated spiral/tiptoe pattern.
#
# alwayslink = True is mandatory: the registrar instance is the only thing
# the binary references; without it the linker strips the TU and the
# algorithm would silently not register.
cc_library(
    name = "pirana_pir_operator",
    srcs = ["pirana_pir.cc"],
    hdrs = ["pirana_pir.h"],
    alwayslink = True,
    deps = [
        "@pirana//:pirana_lib",
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ],
)
