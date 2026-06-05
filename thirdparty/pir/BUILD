load("@rules_cc//cc:defs.bzl", "cc_library")

# Hosts the BUILD wrappers used by WORKSPACE new_git_repository entries that
# pull external PIR algorithm implementations. The wrappers themselves are
# loaded via `build_file = "@//thirdparty/pir:BUILD.spiral"` etc. and live
# alongside this file as `BUILD.<algo>` siblings — exported so bazel can read
# them at workspace-loading time.

package(default_visibility = ["//visibility:public"])

exports_files([
    "BUILD.spiral",
])
