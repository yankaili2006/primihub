"""
Training Logger Executor
学习日志记录执行器

单方学习日志记录任务的执行入口。
"""

import logging
import os
from typing import Any, Dict, Optional

from ..base import LocalBaseModel
from .base import (
    TrainingLogger,
    MetricsTracker,
    LogLevel,
)

logger = logging.getLogger(__name__)


class TrainingLoggerExecutor(LocalBaseModel):
    """
    学习日志记录执行器

    执行单方学习日志记录任务。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_params(self):
        """解析参数"""
        # 操作类型（start, log, end, save）
        self.operation = self.common_params.get("operation", "log")
        # 任务名称
        self.task_name = self.common_params.get("task_name", "training")
        # 会话ID
        self.session_id = self.common_params.get("session_id", None)
        # 日志目录
        self.log_dir = self.common_params.get("log_dir", "./logs")
        # 输出路径
        self.output_path = self.common_params.get("output_path", "")

        # 日志配置
        self.config = self.common_params.get("config", {})

        # 日志消息
        self.message = self.common_params.get("message", "")
        self.log_level = self.common_params.get("log_level", "INFO")
        self.step = self.common_params.get("step", None)
        self.epoch = self.common_params.get("epoch", None)

        # 指标
        self.metrics = self.common_params.get("metrics", {})

        # 检查点路径
        self.checkpoint_path = self.common_params.get("checkpoint_path", "")

        # 会话状态（用于end操作）
        self.status = self.common_params.get("status", "completed")
        self.final_metrics = self.common_params.get("final_metrics", {})

    def run(self) -> Dict[str, Any]:
        """执行日志记录任务"""
        logger.info(f"TrainingLoggerExecutor: Executing {self.operation} operation")

        # 获取或创建日志记录器
        training_logger = self._get_logger()

        if self.operation == "start":
            return self._run_start(training_logger)
        elif self.operation == "log":
            return self._run_log(training_logger)
        elif self.operation == "log_metrics":
            return self._run_log_metrics(training_logger)
        elif self.operation == "log_checkpoint":
            return self._run_log_checkpoint(training_logger)
        elif self.operation == "end":
            return self._run_end(training_logger)
        elif self.operation == "save":
            return self._run_save(training_logger)
        elif self.operation == "get_status":
            return self._run_get_status(training_logger)
        else:
            return {"error": f"Unknown operation: {self.operation}"}

    def _get_logger(self) -> TrainingLogger:
        """获取日志记录器"""
        return TrainingLogger(
            log_dir=self.log_dir,
            task_name=self.task_name,
            session_id=self.session_id,
        )

    def _run_start(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """开始会话"""
        training_logger.start_session(config=self.config)

        # 保存初始状态
        output_path = training_logger.save()

        return {
            "operation": "start",
            "session_id": training_logger.session_id,
            "task_name": training_logger.task_name,
            "log_path": output_path,
            "success": True,
        }

    def _run_log(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """记录日志"""
        level = LogLevel[self.log_level.upper()] if self.log_level else LogLevel.INFO

        training_logger.log(
            level=level,
            message=self.message,
            step=self.step,
            epoch=self.epoch,
            metrics=self.metrics if self.metrics else None,
        )

        return {
            "operation": "log",
            "session_id": training_logger.session_id,
            "message": self.message,
            "level": level.value,
            "success": True,
        }

    def _run_log_metrics(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """记录指标"""
        if not self.metrics:
            return {"error": "No metrics provided"}

        training_logger.log_metrics(
            metrics=self.metrics,
            step=self.step,
            epoch=self.epoch,
        )

        return {
            "operation": "log_metrics",
            "session_id": training_logger.session_id,
            "metrics": self.metrics,
            "step": self.step,
            "epoch": self.epoch,
            "success": True,
        }

    def _run_log_checkpoint(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """记录检查点"""
        if not self.checkpoint_path:
            return {"error": "No checkpoint path provided"}

        training_logger.log_checkpoint(
            checkpoint_path=self.checkpoint_path,
            metrics=self.metrics if self.metrics else None,
            step=self.step,
        )

        return {
            "operation": "log_checkpoint",
            "session_id": training_logger.session_id,
            "checkpoint_path": self.checkpoint_path,
            "success": True,
        }

    def _run_end(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """结束会话"""
        training_logger.end_session(
            status=self.status,
            final_metrics=self.final_metrics if self.final_metrics else None,
        )

        # 保存最终状态
        output_path = training_logger.save(self.output_path if self.output_path else None)

        return {
            "operation": "end",
            "session_id": training_logger.session_id,
            "status": self.status,
            "final_metrics": self.final_metrics,
            "log_path": output_path,
            "success": True,
        }

    def _run_save(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """保存日志"""
        output_path = training_logger.save(self.output_path if self.output_path else None)

        return {
            "operation": "save",
            "session_id": training_logger.session_id,
            "log_path": output_path,
            "success": True,
        }

    def _run_get_status(self, training_logger: TrainingLogger) -> Dict[str, Any]:
        """获取会话状态"""
        session = training_logger.get_session()
        metrics_tracker = training_logger.get_metrics_tracker()

        return {
            "operation": "get_status",
            "session_id": training_logger.session_id,
            "session": session.to_dict() if session else None,
            "latest_metrics": metrics_tracker.get_latest_metrics(),
            "best_metrics": metrics_tracker.get_best_metrics(),
            "log_count": len(training_logger.get_logs()),
            "success": True,
        }
