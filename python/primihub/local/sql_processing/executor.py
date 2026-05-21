"""
SQL Processing Executor
SQL处理执行器

单方SQL处理任务的执行入口。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    SQLEngine,
    SQLValidator,
    SQLQueryBuilder,
)

logger = logging.getLogger(__name__)


class SQLProcessingExecutor(LocalBaseModel):
    """
    SQL处理执行器

    执行单方SQL查询和处理任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # SQL查询语句
        self.query = self.common_params.get("query", "")
        # SQL查询列表（支持多个查询）
        self.queries = self.common_params.get("queries", [])
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # SQL执行后端
        self.backend = self.common_params.get("backend", "sqlite")
        # 主表名（用于注册数据）
        self.table_name = self.common_params.get("table_name", "data")
        # 额外的数据表
        self.extra_tables = self.common_params.get("extra_tables", {})
        # 是否跳过验证
        self.skip_validation = self.common_params.get("skip_validation", False)
        # 是否允许修改操作
        self.allow_modifications = self.common_params.get("allow_modifications", False)

    def run(self) -> Dict[str, Any]:
        """执行SQL处理任务"""
        logger.info("SQLProcessingExecutor: Starting SQL processing")

        # 准备查询列表
        queries = self.queries if self.queries else [self.query] if self.query else []
        if not queries:
            logger.error("No SQL query provided")
            return {"error": "No SQL query provided"}

        # 加载数据
        data, labels = self._load_data()
        if data.empty:
            logger.error("No data loaded")
            return {"error": "No data loaded"}

        logger.info(f"Data loaded: shape={data.shape}")

        # 创建SQL引擎
        engine = SQLEngine(backend=self.backend)

        # 注册主表
        engine.register_table(self.table_name, data)

        # 注册额外的表（如果有）
        for table_name, table_info in self.extra_tables.items():
            if isinstance(table_info, pd.DataFrame):
                engine.register_table(table_name, table_info)
            elif isinstance(table_info, dict) and 'data_path' in table_info:
                # 从文件加载
                try:
                    extra_data = pd.read_csv(table_info['data_path'])
                    engine.register_table(table_name, extra_data)
                except Exception as e:
                    logger.warning(f"Failed to load extra table {table_name}: {e}")

        # 创建验证器
        validator = SQLValidator(allow_modifications=self.allow_modifications)

        # 执行查询
        results = []
        for i, query in enumerate(queries):
            query_result = {
                "query_index": i,
                "query": query,
                "success": False,
            }

            # 验证查询
            if not self.skip_validation:
                validation = validator.validate(query)
                if not validation["valid"]:
                    query_result["error"] = "Validation failed"
                    query_result["validation_errors"] = validation["errors"]
                    results.append(query_result)
                    continue

                if validation["warnings"]:
                    query_result["warnings"] = validation["warnings"]

            # 执行查询
            try:
                result_df = engine.execute(query)
                query_result["success"] = True
                query_result["result_shape"] = {
                    "rows": len(result_df),
                    "columns": len(result_df.columns),
                }
                query_result["columns"] = result_df.columns.tolist()
                query_result["data"] = result_df

            except Exception as e:
                query_result["error"] = str(e)
                logger.error(f"Query {i} failed: {e}")

            results.append(query_result)

        # 准备返回结果
        result = {
            "total_queries": len(queries),
            "successful_queries": sum(1 for r in results if r["success"]),
            "table_info": engine.get_table_info(),
            "results": [{k: v for k, v in r.items() if k != "data"} for r in results],
        }

        # 保存最后一个成功查询的结果
        if self.output_path:
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                last_result = successful_results[-1]["data"]

                if self.output_path.endswith('.csv'):
                    last_result.to_csv(self.output_path, index=False)
                else:
                    self._save_result(last_result, self.output_path)

                result["output_path"] = self.output_path
                logger.info(f"Result saved to: {self.output_path}")

        logger.info("SQLProcessingExecutor: Completed")
        return result
