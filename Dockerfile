# Copyright 2022 PrimiHub
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:20.04 as builder

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN  apt update \
  && apt install -y python3 python3-dev gcc-8 g++-8 python-dev libgmp-dev python3-pip tzdata cmake libmysqlclient-dev chrpath \
  && apt install -y automake ca-certificates git libtool m4 patch pkg-config unzip make wget curl zip ninja-build npm \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
  && rm -rf /var/lib/apt/lists/*

# install  bazelisk
RUN npm install -g @bazel/bazelisk

WORKDIR /src
ADD . /src

# Bazel build primihub-node & primihub-cli & paillier shared library
# Cache mount preserves ~1h Bazel build across image rebuilds
RUN --mount=type=cache,target=/root/.cache/bazel \
  bash pre_build.sh \
  && mv -f WORKSPACE_GITHUB WORKSPACE \
  && make mysql=y \
  && tar zcfh bazel-bin.tar.gz bazel-bin/cli \
        bazel-bin/node \
        bazel-bin/_solib* \
        bazel-bin/task_main \
        python \
        config \
        example \
        data 2>/dev/null || true \
  && if ls bazel-bin/src/primihub/pybind_warpper/*.so 2>/dev/null; then \
       tar zrfh bazel-bin.tar.gz bazel-bin/src/primihub/pybind_warpper/*.so 2>/dev/null || true; \
     fi \
  && if ls bazel-bin/src/primihub/task/pybind_wrapper/*.so 2>/dev/null; then \
       tar zrfh bazel-bin.tar.gz bazel-bin/src/primihub/task/pybind_wrapper/*.so 2>/dev/null || true; \
     fi

FROM ubuntu:20.04 as runner

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/bazel-bin.tar.gz /opt/bazel-bin.tar.gz
COPY --from=builder /src/src/primihub/protos/ /app/src/primihub/protos/

WORKDIR /app

# Copy opt_paillier_c2py.so linkcontext.so to /app/python, this enable setup.py find it.
RUN tar zxf /opt/bazel-bin.tar.gz \
  && mkdir log

RUN ln -s -f bazel-bin/cli primihub-cli
RUN ln -s -f bazel-bin/node primihub-node

WORKDIR /app/python

RUN --mount=type=cache,target=/root/.cache/pip \
  python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
  && python3 -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
  && python3 setup.py install


WORKDIR /app

# gRPC server port
EXPOSE 50050
