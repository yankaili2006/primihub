# Copyright 2024 PrimiHub
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
校验器模块

导出所有校验器类。
"""

from .base import BaseValidator, SelectValidator
from .filter_validator import FilterValidator
from .join_validator import JoinValidator
from .aggregate_validator import AggregateValidator
from .group_by_validator import GroupByValidator
from .order_by_validator import OrderByValidator
from .window_validator import WindowFunctionValidator
from .subquery_validator import SubqueryValidator

__all__ = [
    'BaseValidator',
    'SelectValidator',
    'FilterValidator',
    'JoinValidator',
    'AggregateValidator',
    'GroupByValidator',
    'OrderByValidator',
    'WindowFunctionValidator',
    'SubqueryValidator',
]


# 校验器注册表
VALIDATOR_REGISTRY = {
    'select': SelectValidator,
    'filter': FilterValidator,
    'join': JoinValidator,
    'aggregate': AggregateValidator,
    'group_by': GroupByValidator,
    'order_by': OrderByValidator,
    'window': WindowFunctionValidator,
    'subquery': SubqueryValidator,
}


def get_validator_class(name: str):
    """获取校验器类

    Args:
        name: 校验器名称

    Returns:
        校验器类
    """
    return VALIDATOR_REGISTRY.get(name.lower())


def get_all_validators():
    """获取所有校验器类

    Returns:
        校验器类列表
    """
    return list(VALIDATOR_REGISTRY.values())
