"""
Training Logger Base Classes
学习日志记录基础类

提供训练日志记录和指标跟踪功能。
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: str
    level: str
    message: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class TrainingSession:
    """训练会话"""
    session_id: str
    task_name: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    config: Dict = field(default_factory=dict)
    final_metrics: Dict = field(default_factory=dict)
    logs: List[LogEntry] = field(default_factory=list)
    checkpoints: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "task_name": self.task_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "config": self.config,
            "final_metrics": self.final_metrics,
            "logs": [log.to_dict() if isinstance(log, LogEntry) else log for log in self.logs],
            "checkpoints": self.checkpoints,
        }


class MetricsTracker:
    """
    指标跟踪器

    跟踪训练过程中的各种指标。
    """

    def __init__(self):
        self._metrics_history = {}
        self._current_step = 0
        self._current_epoch = 0
        self._best_metrics = {}

    def log_metrics(self, metrics: Dict[str, float],
                    step: Optional[int] = None,
                    epoch: Optional[int] = None):
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 当前步数
            epoch: 当前轮次
        """
        timestamp = datetime.now().isoformat()

        if step is not None:
            self._current_step = step
        if epoch is not None:
            self._current_epoch = epoch

        for name, value in metrics.items():
            if name not in self._metrics_history:
                self._metrics_history[name] = []

            self._metrics_history[name].append({
                "value": value,
                "step": self._current_step,
                "epoch": self._current_epoch,
                "timestamp": timestamp,
            })

            # 更新最佳指标
            if name not in self._best_metrics:
                self._best_metrics[name] = {"value": value, "step": self._current_step}
            else:
                # 假设指标越大越好（损失类指标需要取反）
                if "loss" in name.lower() or "error" in name.lower():
                    if value < self._best_metrics[name]["value"]:
                        self._best_metrics[name] = {"value": value, "step": self._current_step}
                else:
                    if value > self._best_metrics[name]["value"]:
                        self._best_metrics[name] = {"value": value, "step": self._current_step}

    def get_metric_history(self, name: str) -> List[Dict]:
        """获取指标历史"""
        return self._metrics_history.get(name, [])

    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """获取所有指标历史"""
        return self._metrics_history

    def get_best_metrics(self) -> Dict[str, Dict]:
        """获取最佳指标"""
        return self._best_metrics

    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的指标值"""
        latest = {}
        for name, history in self._metrics_history.items():
            if history:
                latest[name] = history[-1]["value"]
        return latest

    def reset(self):
        """重置跟踪器"""
        self._metrics_history = {}
        self._current_step = 0
        self._current_epoch = 0
        self._best_metrics = {}


class TrainingLogger:
    """
    训练日志记录器

    记录和管理训练过程的日志。
    """

    def __init__(self, log_dir: str = "./logs",
                 task_name: str = "training",
                 session_id: Optional[str] = None):
        """
        初始化训练日志记录器

        Args:
            log_dir: 日志目录
            task_name: 任务名称
            session_id: 会话ID
        """
        self.log_dir = log_dir
        self.task_name = task_name
        self.session_id = session_id or self._generate_session_id()

        self._metrics_tracker = MetricsTracker()
        self._session = None
        self._log_entries = []

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def start_session(self, config: Optional[Dict] = None):
        """开始训练会话"""
        self._session = TrainingSession(
            session_id=self.session_id,
            task_name=self.task_name,
            start_time=datetime.now().isoformat(),
            config=config or {},
        )
        self.log(LogLevel.INFO, f"Training session started: {self.session_id}")
        logger.info(f"Training session started: {self.session_id}")

    def end_session(self, status: str = "completed", final_metrics: Optional[Dict] = None):
        """结束训练会话"""
        if self._session:
            self._session.end_time = datetime.now().isoformat()
            self._session.status = status
            self._session.final_metrics = final_metrics or self._metrics_tracker.get_latest_metrics()
            self._session.logs = self._log_entries

        self.log(LogLevel.INFO, f"Training session ended: {status}")
        logger.info(f"Training session ended: {self.session_id}, status={status}")

    def log(self, level: Union[LogLevel, str], message: str,
            step: Optional[int] = None, epoch: Optional[int] = None,
            metrics: Optional[Dict] = None, extra: Optional[Dict] = None):
        """
        记录日志

        Args:
            level: 日志级别
            message: 日志消息
            step: 当前步数
            epoch: 当前轮次
            metrics: 相关指标
            extra: 额外信息
        """
        if isinstance(level, LogLevel):
            level = level.value

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
        )
        self._log_entries.append(entry)

        # 同时记录到Python logger
        log_level = getattr(logging, level, logging.INFO)
        logger.log(log_level, f"[{self.session_id}] {message}")

    def log_metrics(self, metrics: Dict[str, float],
                    step: Optional[int] = None,
                    epoch: Optional[int] = None):
        """记录指标"""
        self._metrics_tracker.log_metrics(metrics, step, epoch)
        self.log(LogLevel.INFO, f"Metrics logged: {metrics}", step=step, epoch=epoch, metrics=metrics)

    def log_checkpoint(self, checkpoint_path: str,
                       metrics: Optional[Dict] = None,
                       step: Optional[int] = None):
        """记录检查点"""
        checkpoint_info = {
            "path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or self._metrics_tracker.get_latest_metrics(),
            "step": step,
        }
        if self._session:
            self._session.checkpoints.append(checkpoint_info)

        self.log(LogLevel.INFO, f"Checkpoint saved: {checkpoint_path}", step=step)

    def get_metrics_tracker(self) -> MetricsTracker:
        """获取指标跟踪器"""
        return self._metrics_tracker

    def get_session(self) -> Optional[TrainingSession]:
        """获取当前会话"""
        return self._session

    def get_logs(self) -> List[LogEntry]:
        """获取所有日志条目"""
        return self._log_entries

    def save(self, output_path: Optional[str] = None):
        """
        保存日志到文件

        Args:
            output_path: 输出路径
        """
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"{self.session_id}.json")

        session_dict = self._session.to_dict() if self._session else {
            "session_id": self.session_id,
            "task_name": self.task_name,
            "logs": [log.to_dict() if isinstance(log, LogEntry) else log for log in self._log_entries],
        }

        # 添加指标历史
        session_dict["metrics_history"] = self._metrics_tracker.get_all_metrics()
        session_dict["best_metrics"] = self._metrics_tracker.get_best_metrics()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Training logs saved to: {output_path}")
        return output_path

    @classmethod
    def load(cls, path: str) -> 'TrainingLogger':
        """从文件加载日志"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trainer_logger = cls(
            task_name=data.get("task_name", "training"),
            session_id=data.get("session_id"),
        )

        # 恢复日志条目
        for log_data in data.get("logs", []):
            entry = LogEntry(**log_data)
            trainer_logger._log_entries.append(entry)

        # 恢复会话
        if "start_time" in data:
            trainer_logger._session = TrainingSession(
                session_id=data["session_id"],
                task_name=data["task_name"],
                start_time=data["start_time"],
                end_time=data.get("end_time"),
                status=data.get("status", "unknown"),
                config=data.get("config", {}),
                final_metrics=data.get("final_metrics", {}),
            )

        return trainer_logger
