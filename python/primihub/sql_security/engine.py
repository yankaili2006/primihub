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
SQL安全校验引擎

主校验引擎，协调所有校验器进行SQL安全检查。
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from .field_security import TableSecurityConfig
from .validation_result import (
    ValidationResult,
    ValidationIssue,
    ValidationIssueType,
    RiskLevel
)
from .security_policy import (
    SecurityPolicy,
    SchemaSecurityManager,
    ValidationContext,
    SecurityConfigLoader
)
from .sql_parser import SQLParser, ParsedSQL
from .validators import (
    BaseValidator,
    SelectValidator,
    FilterValidator,
    JoinValidator,
    AggregateValidator,
    GroupByValidator,
    OrderByValidator,
    WindowFunctionValidator,
    SubqueryValidator
)


class SQLSecurityEngine:
    """SQL安全校验引擎

    协调所有校验器进行SQL安全检查。

    Usage:
        engine = SQLSecurityEngine()
        engine.load_config("config/sql_security_config.yaml")
        result = engine.validate("SELECT * FROM users WHERE salary > 50000")
        if not result.is_valid:
            for issue in result.issues:
                print(f"[{issue.risk_level.name}] {issue.description}")
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化引擎

        Args:
            config_path: 可选的配置文件路径
        """
        self.policy = SecurityPolicy()
        self.schema_manager = SchemaSecurityManager()
        self._sql_parser: Optional[SQLParser] = None
        self._initialized = False

        if config_path:
            self.load_config(config_path)

    def _get_parser(self) -> SQLParser:
        """获取SQL解析器（延迟初始化）"""
        if self._sql_parser is None:
            self._sql_parser = SQLParser()
        return self._sql_parser

    def load_config(self, config_path: str) -> None:
        """从配置文件加载安全配置

        Args:
            config_path: YAML配置文件路径
        """
        self.policy, self.schema_manager = SecurityConfigLoader.load(config_path=config_path)
        self._initialized = True

    def load_config_from_dict(self, config: Dict[str, Any]) -> None:
        """从字典加载配置

        Args:
            config: 配置字典
        """
        self.policy, self.schema_manager = SecurityConfigLoader.load(config_dict=config)
        self._initialized = True

    def register_table(self, config: TableSecurityConfig) -> None:
        """注册表安全配置

        Args:
            config: 表安全配置
        """
        self.schema_manager.register_table(config)

    def update_policy(self, policy: SecurityPolicy) -> None:
        """更新安全策略

        Args:
            policy: 新的安全策略
        """
        self.policy = policy

    def validate(self, sql: str, current_party: str = "default") -> ValidationResult:
        """校验SQL语句

        Args:
            sql: SQL语句
            current_party: 当前执行方

        Returns:
            ValidationResult 校验结果
        """
        result = ValidationResult(original_sql=sql, current_party=current_party)

        try:
            # 解析SQL
            parser = self._get_parser()
            parsed_sql = parser.parse(sql)

            if not parsed_sql.is_valid:
                result.add_issue(ValidationIssue(
                    issue_type=ValidationIssueType.SQL_SYNTAX_ERROR,
                    risk_level=RiskLevel.HIGH,
                    description=f"SQL解析错误: {parsed_sql.parse_error}",
                    sql_fragment=sql[:100]
                ))
                result.is_valid = False
                return result

            # 注册表别名
            self.schema_manager.clear_aliases()
            for alias, table_name in parsed_sql.table_aliases.items():
                self.schema_manager.register_alias(alias, table_name)

            # 创建校验上下文
            context = ValidationContext(
                schema_manager=self.schema_manager,
                policy=self.policy,
                current_party=current_party
            )

            # 获取需要运行的校验器
            validators = self._create_validators(parsed_sql, context)

            # 运行所有校验器
            for validator in validators:
                try:
                    issues = validator.validate(parsed_sql)
                    for issue in issues:
                        result.add_issue(issue)
                except Exception as e:
                    result.add_issue(ValidationIssue(
                        issue_type=ValidationIssueType.UNKNOWN_RISK,
                        risk_level=RiskLevel.LOW,
                        description=f"校验器 {validator.get_name()} 执行错误: {str(e)}",
                        validator_name=validator.get_name()
                    ))

            # 检查未知表
            self._check_unknown_tables(parsed_sql, result)

        except Exception as e:
            result.add_issue(ValidationIssue(
                issue_type=ValidationIssueType.UNKNOWN_RISK,
                risk_level=RiskLevel.MEDIUM,
                description=f"校验过程发生错误: {str(e)}",
                sql_fragment=sql[:100]
            ))

        return result

    def _create_validators(self, parsed_sql: ParsedSQL, context: ValidationContext) -> List[BaseValidator]:
        """根据SQL内容创建需要的校验器

        Args:
            parsed_sql: 解析后的SQL
            context: 校验上下文

        Returns:
            校验器列表
        """
        validators: List[BaseValidator] = []

        # SELECT校验器（总是需要）
        validators.append(SelectValidator(context))

        # 根据SQL内容添加其他校验器

        # WHERE/HAVING条件校验
        if parsed_sql.where_conditions or parsed_sql.where_raw or \
           parsed_sql.having_conditions or parsed_sql.having_raw:
            validators.append(FilterValidator(context))

        # JOIN校验
        if parsed_sql.has_joins():
            validators.append(JoinValidator(context))

        # 聚合函数校验
        if parsed_sql.has_aggregates():
            validators.append(AggregateValidator(context))

        # GROUP BY校验
        if parsed_sql.has_group_by():
            validators.append(GroupByValidator(context))

        # ORDER BY校验
        if parsed_sql.has_order_by():
            validators.append(OrderByValidator(context))

        # 窗口函数校验
        if parsed_sql.has_window_functions():
            validators.append(WindowFunctionValidator(context))

        # 子查询校验
        if parsed_sql.has_subqueries():
            validators.append(SubqueryValidator(context))

        return validators

    def _check_unknown_tables(self, parsed_sql: ParsedSQL, result: ValidationResult) -> None:
        """检查未知表

        Args:
            parsed_sql: 解析后的SQL
            result: 校验结果
        """
        for table in parsed_sql.from_tables:
            if not self.schema_manager.has_table(table.table_name):
                result.add_issue(ValidationIssue(
                    issue_type=ValidationIssueType.UNKNOWN_TABLE,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"表 {table.table_name} 未在安全配置中注册",
                    sql_fragment=table.table_name,
                    remediation="请在配置文件中添加该表的安全配置，或使用默认的PROTECTED级别",
                    affected_tables=[table.table_name]
                ))

    def validate_batch(self, sqls: List[str], current_party: str = "default") -> List[ValidationResult]:
        """批量校验SQL语句

        Args:
            sqls: SQL语句列表
            current_party: 当前执行方

        Returns:
            校验结果列表
        """
        return [self.validate(sql, current_party) for sql in sqls]

    def is_sql_safe(self, sql: str, current_party: str = "default") -> bool:
        """快速检查SQL是否安全

        Args:
            sql: SQL语句
            current_party: 当前执行方

        Returns:
            是否安全
        """
        result = self.validate(sql, current_party)
        return result.is_valid

    def get_sql_risk_level(self, sql: str, current_party: str = "default") -> RiskLevel:
        """获取SQL的风险等级

        Args:
            sql: SQL语句
            current_party: 当前执行方

        Returns:
            风险等级
        """
        result = self.validate(sql, current_party)
        return result.overall_risk_level


def create_engine(config_path: Optional[str] = None,
                  config_dict: Optional[Dict[str, Any]] = None) -> SQLSecurityEngine:
    """创建校验引擎的便捷函数

    Args:
        config_path: 配置文件路径
        config_dict: 配置字典

    Returns:
        SQLSecurityEngine 实例
    """
    engine = SQLSecurityEngine()
    if config_path:
        engine.load_config(config_path)
    elif config_dict:
        engine.load_config_from_dict(config_dict)
    return engine
