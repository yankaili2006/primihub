#!/bin/bash
set -x
pids=$(ps -ef |grep fusion-simple.jar |grep ${USER} |grep -v grep |awk '{print $2}')

if [ -n "${pids}" ]; then
  kill -9 ${pids}
fi
