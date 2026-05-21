"""
Log Exporter Executor
日志导出执行器

单方学习日志导出任务的执行入口。
"""

import logging
import os
from typing import Any, Dict, Optional

from ..base import LocalBaseModel
from .base import (
    LogExporter,
    JSONExporter,
    CSVExporter,
    HTMLExporter,
    TensorBoardExporter,
)

logger = logging.getLogger(__name__)


class LogExporterExecutor(LocalBaseModel):
    """
    日志导出执行器

    执行单方学习日志导出任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 导出格式
        self.format = self.common_params.get("format", "json")
        # 支持多种格式同时导出
        self.formats = self.common_params.get("formats", [])
        # 输入日志文件路径
        self.log_path = self.common_params.get("log_path", "")
        # 直接传入的日志数据
        self.log_data = self.common_params.get("log_data", {})
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")
        # 输出目录
        self.output_dir = self.common_params.get("output_dir", "./exports")

        # JSON导出参数
        self.json_indent = self.common_params.get("json_indent", 2)

        # CSV导出参数
        self.export_logs = self.common_params.get("export_logs", True)
        self.export_metrics = self.common_params.get("export_metrics", True)

        # HTML导出参数
        self.include_charts = self.common_params.get("include_charts", True)

    def run(self) -> Dict[str, Any]:
        """执行日志导出任务"""
        logger.info(f"LogExporterExecutor: Starting log export in format={self.format}")

        # 获取日志数据
        log_data = self._get_log_data()
        if not log_data:
            logger.error("No log data available")
            return {"error": "No log data available"}

        # 确定要导出的格式
        formats = self.formats if self.formats else [self.format]

        # 执行导出
        results = {}
        for fmt in formats:
            try:
                output_path = self._export_format(fmt, log_data)
                results[fmt] = {
                    "success": True,
                    "output_path": output_path,
                }
            except Exception as e:
                logger.error(f"Failed to export in format {fmt}: {e}")
                results[fmt] = {
                    "success": False,
                    "error": str(e),
                }

        result = {
            "formats": formats,
            "results": results,
            "successful_exports": sum(1 for r in results.values() if r.get("success")),
            "total_exports": len(formats),
        }

        logger.info(f"LogExporterExecutor: Completed. {result['successful_exports']}/{result['total_exports']} exports successful")
        return result

    def _get_log_data(self) -> Dict:
        """获取日志数据"""
        # 优先使用直接传入的数据
        if self.log_data:
            return self.log_data

        # 从文件加载
        if self.log_path and os.path.exists(self.log_path):
            return LogExporter.load_log_file(self.log_path)

        return {}

    def _export_format(self, fmt: str, log_data: Dict) -> str:
        """导出指定格式"""
        # 确定输出路径
        session_id = log_data.get('session_id', 'training_log')
        output_path = self.output_path or os.path.join(self.output_dir, session_id)

        # 创建导出器
        exporters = {
            "json": JSONExporter(indent=self.json_indent),
            "csv": CSVExporter(
                export_logs=self.export_logs,
                export_metrics=self.export_metrics,
            ),
            "html": HTMLExporter(include_charts=self.include_charts),
            "tensorboard": TensorBoardExporter(),
        }

        if fmt not in exporters:
            raise ValueError(f"Unknown format: {fmt}")

        exporter = exporters[fmt]

        # 调整输出路径
        if fmt == "json":
            final_path = output_path if output_path.endswith('.json') else f"{output_path}.json"
        elif fmt == "html":
            final_path = output_path if output_path.endswith('.html') else f"{output_path}.html"
        elif fmt == "csv":
            final_path = output_path if os.path.isdir(output_path) else os.path.dirname(output_path) or self.output_dir
        elif fmt == "tensorboard":
            final_path = output_path if os.path.isdir(output_path) else os.path.join(self.output_dir, f"{session_id}_tensorboard")
        else:
            final_path = output_path

        # 确保输出目录存在
        dir_path = os.path.dirname(final_path) if not os.path.isdir(final_path) else final_path
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 执行导出
        return exporter.export(log_data, final_path)
