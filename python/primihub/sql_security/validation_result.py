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
校验结果定义模块

定义SQL安全校验的结果结构，包括风险等级、问题类型和详细问题描述。
"""

import json
from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


class RiskLevel(IntEnum):
    """风险等级枚举

    定义五个风险等级，数值越大风险越高:
    - SAFE: 安全，无风险
    - LOW: 低风险，可以执行但需注意
    - MEDIUM: 中等风险，建议审查
    - HIGH: 高风险，强烈建议不执行
    - CRITICAL: 严重风险，禁止执行
    """
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def is_acceptable(self) -> bool:
        """检查风险是否可接受 (LOW 及以下)"""
        return self <= RiskLevel.LOW

    def requires_review(self) -> bool:
        """检查是否需要审查 (MEDIUM 及以上)"""
        return self >= RiskLevel.MEDIUM

    def should_block(self) -> bool:
        """检查是否应该阻止执行 (HIGH 及以上)"""
        return self >= RiskLevel.HIGH

    @classmethod
    def from_string(cls, level_str: str) -> 'RiskLevel':
        """从字符串转换"""
        level_map = {
            'SAFE': cls.SAFE,
            'LOW': cls.LOW,
            'MEDIUM': cls.MEDIUM,
            'HIGH': cls.HIGH,
            'CRITICAL': cls.CRITICAL
        }
        return level_map.get(level_str.upper(), cls.MEDIUM)


class ValidationIssueType(Enum):
    """校验问题类型枚举

    定义各种可能的安全问题类型。
    """
    # 字段暴露相关
    DIRECT_FIELD_EXPOSURE = "直接字段暴露"
    SELECT_STAR_FORBIDDEN = "禁止使用SELECT *"

    # k-匿名相关
    INSUFFICIENT_K_ANONYMITY = "k-匿名不足"
    SMALL_GROUP_RISK = "小分组泄露风险"

    # 过滤条件相关
    SENSITIVE_FILTER_CONDITION = "敏感过滤条件"
    EXACT_MATCH_ON_PRIVATE = "私有字段精确匹配"
    NARROW_RANGE_FILTER = "范围过滤过窄"
    LIKE_PATTERN_TOO_SPECIFIC = "LIKE模式过于精确"

    # JOIN相关
    UNSAFE_JOIN = "不安全的JOIN"
    CROSS_PARTY_JOIN_FORBIDDEN = "禁止跨方JOIN"
    JOIN_ON_SENSITIVE_FIELD = "敏感字段JOIN"
    JOIN_TYPE_LEAKAGE = "JOIN类型泄露存在性"

    # 聚合相关
    AGGREGATE_DISCLOSURE = "聚合披露风险"
    BOUNDARY_VALUE_LEAKAGE = "边界值泄露"
    ATTRIBUTE_INFERENCE = "属性推断风险"
    COUNT_DISCLOSURE = "计数泄露"

    # 分组相关
    GROUP_BY_DISCLOSURE = "分组披露风险"
    GROUP_BY_ON_SENSITIVE = "敏感字段分组"
    QUASI_IDENTIFIER_RISK = "准标识符风险"
    FINE_GRAINED_GROUPING = "分组粒度过细"

    # 排序相关
    PRIVACY_LEAKAGE_VIA_ORDER = "排序泄露隐私"
    TOP_N_LEAKAGE = "TOP-N泄露风险"
    ORDER_BY_SENSITIVE = "敏感字段排序"

    # 窗口函数相关
    WINDOW_FUNCTION_RISK = "窗口函数风险"
    WINDOW_FUNCTION_FORBIDDEN = "禁止使用窗口函数"
    RANKING_FUNCTION_RISK = "排名函数风险"
    ADJACENT_VALUE_LEAKAGE = "邻近值泄露"

    # 子查询相关
    SUBQUERY_RISK = "子查询风险"
    CORRELATED_SUBQUERY_FORBIDDEN = "禁止关联子查询"
    EXISTS_LEAKAGE = "EXISTS存在性泄露"
    IN_SUBQUERY_MEMBERSHIP = "IN子查询成员推断"
    SCALAR_SUBQUERY_RISK = "标量子查询风险"

    # 其他
    UNKNOWN_RISK = "未知风险"
    SQL_SYNTAX_ERROR = "SQL语法错误"
    UNKNOWN_TABLE = "未知表"
    UNKNOWN_FIELD = "未知字段"


@dataclass
class ValidationIssue:
    """单个校验问题

    描述发现的具体安全问题。

    Attributes:
        issue_type: 问题类型
        risk_level: 风险等级
        description: 问题描述
        sql_fragment: 相关的SQL片段
        remediation: 建议的修复方案
        affected_fields: 受影响的字段列表
        affected_tables: 受影响的表列表
        validator_name: 发现问题的校验器名称
    """
    issue_type: ValidationIssueType
    risk_level: RiskLevel
    description: str
    sql_fragment: str = ""
    remediation: str = ""
    affected_fields: List[str] = field(default_factory=list)
    affected_tables: List[str] = field(default_factory=list)
    validator_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'issue_type': self.issue_type.name,
            'issue_type_desc': self.issue_type.value,
            'risk_level': self.risk_level.name,
            'risk_level_value': int(self.risk_level),
            'description': self.description,
            'sql_fragment': self.sql_fragment,
            'remediation': self.remediation,
            'affected_fields': self.affected_fields,
            'affected_tables': self.affected_tables,
            'validator_name': self.validator_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationIssue':
        """从字典创建"""
        return cls(
            issue_type=ValidationIssueType[data.get('issue_type', 'UNKNOWN_RISK')],
            risk_level=RiskLevel.from_string(data.get('risk_level', 'MEDIUM')),
            description=data.get('description', ''),
            sql_fragment=data.get('sql_fragment', ''),
            remediation=data.get('remediation', ''),
            affected_fields=data.get('affected_fields', []),
            affected_tables=data.get('affected_tables', []),
            validator_name=data.get('validator_name', '')
        )


@dataclass
class ValidationResult:
    """校验结果

    SQL安全校验的完整结果。

    Attributes:
        is_valid: 是否通过校验
        overall_risk_level: 整体风险等级
        issues: 具体问题列表
        original_sql: 原始SQL语句
        sanitized_sql: 清理后的SQL（如果可能）
        validation_time: 校验时间
        current_party: 当前执行方
        requires_mpc: 是否需要MPC计算
    """
    is_valid: bool = True
    overall_risk_level: RiskLevel = RiskLevel.SAFE
    issues: List[ValidationIssue] = field(default_factory=list)
    original_sql: str = ""
    sanitized_sql: Optional[str] = None
    validation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    current_party: str = "default"
    requires_mpc: bool = False

    def add_issue(self, issue: ValidationIssue) -> None:
        """添加问题并更新整体风险等级

        Args:
            issue: 校验问题
        """
        self.issues.append(issue)

        # 更新整体风险等级（取最高）
        if issue.risk_level > self.overall_risk_level:
            self.overall_risk_level = issue.risk_level

        # 如果有 HIGH 及以上的问题，标记为无效
        if issue.risk_level >= RiskLevel.HIGH:
            self.is_valid = False

    def has_critical_issues(self) -> bool:
        """检查是否有严重问题"""
        return self.overall_risk_level == RiskLevel.CRITICAL

    def has_high_risk_issues(self) -> bool:
        """检查是否有高风险问题"""
        return self.overall_risk_level >= RiskLevel.HIGH

    def get_issues_by_type(self, issue_type: ValidationIssueType) -> List[ValidationIssue]:
        """按类型获取问题"""
        return [i for i in self.issues if i.issue_type == issue_type]

    def get_issues_by_risk_level(self, risk_level: RiskLevel) -> List[ValidationIssue]:
        """按风险等级获取问题"""
        return [i for i in self.issues if i.risk_level == risk_level]

    def get_high_risk_issues(self) -> List[ValidationIssue]:
        """获取高风险及以上的问题"""
        return [i for i in self.issues if i.risk_level >= RiskLevel.HIGH]

    def get_affected_fields(self) -> List[str]:
        """获取所有受影响的字段"""
        fields = set()
        for issue in self.issues:
            fields.update(issue.affected_fields)
        return list(fields)

    def get_affected_tables(self) -> List[str]:
        """获取所有受影响的表"""
        tables = set()
        for issue in self.issues:
            tables.update(issue.affected_tables)
        return list(tables)

    def merge(self, other: 'ValidationResult') -> None:
        """合并另一个校验结果

        Args:
            other: 另一个校验结果
        """
        for issue in other.issues:
            self.add_issue(issue)

        if other.requires_mpc:
            self.requires_mpc = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'overall_risk_level': self.overall_risk_level.name,
            'overall_risk_level_value': int(self.overall_risk_level),
            'issues': [i.to_dict() for i in self.issues],
            'issues_count': len(self.issues),
            'original_sql': self.original_sql,
            'sanitized_sql': self.sanitized_sql,
            'validation_time': self.validation_time,
            'current_party': self.current_party,
            'requires_mpc': self.requires_mpc,
            'affected_fields': self.get_affected_fields(),
            'affected_tables': self.get_affected_tables()
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """从字典创建"""
        result = cls(
            is_valid=data.get('is_valid', True),
            overall_risk_level=RiskLevel.from_string(data.get('overall_risk_level', 'SAFE')),
            original_sql=data.get('original_sql', ''),
            sanitized_sql=data.get('sanitized_sql'),
            validation_time=data.get('validation_time', datetime.now().isoformat()),
            current_party=data.get('current_party', 'default'),
            requires_mpc=data.get('requires_mpc', False)
        )

        for issue_data in data.get('issues', []):
            result.issues.append(ValidationIssue.from_dict(issue_data))

        return result

    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationResult':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """字符串表示"""
        status = "PASS" if self.is_valid else "FAIL"
        return (f"ValidationResult({status}, "
                f"risk={self.overall_risk_level.name}, "
                f"issues={len(self.issues)})")

    def summary(self) -> str:
        """生成简要摘要"""
        lines = [
            f"SQL安全校验结果: {'通过' if self.is_valid else '未通过'}",
            f"整体风险等级: {self.overall_risk_level.name}",
            f"发现问题数: {len(self.issues)}"
        ]

        if self.issues:
            lines.append("\n主要问题:")
            for i, issue in enumerate(self.issues[:5], 1):  # 最多显示5个
                lines.append(f"  {i}. [{issue.risk_level.name}] {issue.description}")

            if len(self.issues) > 5:
                lines.append(f"  ... 还有 {len(self.issues) - 5} 个问题")

        return "\n".join(lines)
