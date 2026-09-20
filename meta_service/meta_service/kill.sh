#!/bin/bash
set -x
pids=$(ps -ef |grep meta-server.jar |grep ${USER} |grep -v grep |awk '{print $2}')

if [ -n "${pids}" ]; then
  kill -9 ${pids}
fi
