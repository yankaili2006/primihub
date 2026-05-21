"""
Python Script Executor
Python脚本执行器

单方Python脚本任务的执行入口。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base import LocalBaseModel
from .base import (
    ScriptRunner,
    ScriptValidator,
    ScriptContext,
)

logger = logging.getLogger(__name__)


class PythonScriptExecutor(LocalBaseModel):
    """
    Python脚本执行器

    执行单方自定义Python脚本任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # Python脚本内容
        self.script = self.common_params.get("script", "")
        # 脚本文件路径（优先级低于script）
        self.script_path = self.common_params.get("script_path", "")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 自定义参数
        self.script_params = self.common_params.get("script_params", {})
        # 是否允许文件操作
        self.allow_file_ops = self.common_params.get("allow_file_ops", False)
        # 是否允许系统调用
        self.allow_system_calls = self.common_params.get("allow_system_calls", False)
        # 执行超时时间
        self.timeout = self.common_params.get("timeout", 60)
        # 是否跳过验证
        self.skip_validation = self.common_params.get("skip_validation", False)

    def run(self) -> Dict[str, Any]:
        """执行Python脚本任务"""
        logger.info("PythonScriptExecutor: Starting Python script execution")

        # 获取脚本内容
        script = self._get_script()
        if not script:
            logger.error("No script provided")
            return {"error": "No script provided"}

        # 加载数据
        data, labels = self._load_data()
        logger.info(f"Data loaded: shape={data.shape if not data.empty else 'empty'}")

        # 验证脚本
        if not self.skip_validation:
            validator = ScriptValidator(
                allow_file_ops=self.allow_file_ops,
                allow_system_calls=self.allow_system_calls,
            )
            validation_result = validator.validate(script)

            if not validation_result["valid"]:
                logger.error(f"Script validation failed: {validation_result['errors']}")
                return {
                    "error": "Script validation failed",
                    "validation_errors": validation_result["errors"],
                }

            if validation_result["warnings"]:
                logger.warning(f"Script warnings: {validation_result['warnings']}")

        # 创建执行上下文
        context = ScriptContext(
            data=data if not data.empty else None,
            labels=labels,
            params=self.script_params,
        )

        # 执行脚本
        runner = ScriptRunner(timeout=self.timeout)
        execution_result = runner.run(script, context)

        result = {
            "success": execution_result["success"],
            "result": execution_result["result"],
            "output": execution_result["output"],
            "logs": execution_result["logs"],
            "stdout": execution_result["stdout"],
            "stderr": execution_result["stderr"],
        }

        if not execution_result["success"]:
            result["error"] = execution_result["error"]
            logger.error(f"Script execution failed: {execution_result['error']}")
            return result

        # 保存结果
        if self.output_path and context.result is not None:
            if isinstance(context.result, pd.DataFrame):
                if self.output_path.endswith('.csv'):
                    context.result.to_csv(self.output_path, index=False)
                else:
                    self._save_result(context.result, self.output_path)
            else:
                self._save_result(context.result, self.output_path)

            result["output_path"] = self.output_path
            logger.info(f"Result saved to: {self.output_path}")

        logger.info("PythonScriptExecutor: Completed")
        return result

    def _get_script(self) -> str:
        """获取脚本内容"""
        if self.script:
            return self.script

        if self.script_path:
            try:
                with open(self.script_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read script file: {e}")
                return ""

        return ""
