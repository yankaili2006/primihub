load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# PIRANA — Faster Multi-query PIR via Constant-weight Codes
# (zju-abclab/PIRANA, eprint 2022/1401). Vendored protocol sources
# (pir_parms / client / server / utils) from external/PIRANA; the demo
# main.cc / test.cc are NOT vendored — primihub integrates the protocol
# through the pirana_pir operator (src/primihub/kernel/pir/operator/pirana_pir)
# instead.
#
# Requires: Microsoft SEAL (>=4.0) + Microsoft Kuku (2.1.0). Both are
# already declared in the primihub WORKSPACE:
#   @com_github_primihub_seal_40//:seal  (primihub SEAL fork, 4.0)
#   @com_microsoft_kuku//:kuku           (Kuku 2.1.0)
# PIRANA's SEAL usage matches the primihub fork (SEAL 4.0 API surface,
# SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF already set in BUILD.seal —
# required by multiply_mid_form on transparent ciphertexts).

cc_library(
    name = "pirana_lib",
    srcs = [
        "client.cc",
        "pir_parms.cc",
        "server.cc",
        "utils.cc",
    ],
    hdrs = [
        "client.h",
        "pir_parms.h",
        "server.h",
        "utils.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_primihub_seal_40//:seal",
        "@com_microsoft_kuku//:kuku",
    ],
)
