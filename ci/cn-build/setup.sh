#!/bin/bash
# Apply the China build config (gitee git mirrors + OSS distdir) so bazel can
# build behind the GFW. Run from the repo root on the self-hosted .50 runner.
set -e
cp ci/cn-build/WORKSPACE.cn WORKSPACE
cp ci/cn-build/repository_deps.cn.bzl bazel/repository_deps.bzl
[ -f ci/cn-build/WORKSPACE_GITHUB.cn ] && cp ci/cn-build/WORKSPACE_GITHUB.cn WORKSPACE_GITHUB || true
# distdir (http_archive tarballs) — reuse the runner's warm /tmp/ph_distdir
# (populated from oss://primihub/tools + /repository_deps with sha256 symlinks).
mkdir -p /tmp/ph_distdir
echo "CN build config applied; distdir files: $(ls /tmp/ph_distdir 2>/dev/null | wc -l)"
