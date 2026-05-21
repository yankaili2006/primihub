"""
Log Exporter Base Classes
日志导出基础类

提供多种格式的日志导出功能。
"""

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LogExporter(ABC):
    """日志导出器基类"""

    @abstractmethod
    def export(self, log_data: Dict, output_path: str) -> str:
        """
        导出日志

        Args:
            log_data: 日志数据
            output_path: 输出路径

        Returns:
            实际输出路径
        """
        pass

    @staticmethod
    def load_log_file(path: str) -> Dict:
        """加载日志文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class JSONExporter(LogExporter):
    """
    JSON导出器

    将日志导出为JSON格式。
    """

    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        """
        初始化JSON导出器

        Args:
            indent: 缩进空格数
            ensure_ascii: 是否转义非ASCII字符
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def export(self, log_data: Dict, output_path: str) -> str:
        """导出为JSON"""
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=self.indent, ensure_ascii=self.ensure_ascii, default=str)

        logger.info(f"Exported to JSON: {output_path}")
        return output_path


class CSVExporter(LogExporter):
    """
    CSV导出器

    将日志导出为CSV格式。
    """

    def __init__(self, export_logs: bool = True,
                 export_metrics: bool = True):
        """
        初始化CSV导出器

        Args:
            export_logs: 是否导出日志条目
            export_metrics: 是否导出指标历史
        """
        self.export_logs = export_logs
        self.export_metrics = export_metrics

    def export(self, log_data: Dict, output_path: str) -> str:
        """导出为CSV"""
        output_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(output_path).replace('.csv', '')
        if not base_name:
            base_name = log_data.get('session_id', 'training_log')

        exported_files = []

        # 导出日志条目
        if self.export_logs and 'logs' in log_data:
            logs_path = os.path.join(output_dir or '.', f"{base_name}_logs.csv")
            self._export_logs_to_csv(log_data['logs'], logs_path)
            exported_files.append(logs_path)

        # 导出指标历史
        if self.export_metrics and 'metrics_history' in log_data:
            metrics_path = os.path.join(output_dir or '.', f"{base_name}_metrics.csv")
            self._export_metrics_to_csv(log_data['metrics_history'], metrics_path)
            exported_files.append(metrics_path)

        # 导出会话摘要
        summary_path = os.path.join(output_dir or '.', f"{base_name}_summary.csv")
        self._export_summary_to_csv(log_data, summary_path)
        exported_files.append(summary_path)

        logger.info(f"Exported to CSV: {exported_files}")
        return output_dir or '.'

    def _export_logs_to_csv(self, logs: List[Dict], path: str):
        """导出日志条目到CSV"""
        if not logs:
            return

        df = pd.DataFrame(logs)
        df.to_csv(path, index=False)

    def _export_metrics_to_csv(self, metrics_history: Dict, path: str):
        """导出指标历史到CSV"""
        if not metrics_history:
            return

        rows = []
        for metric_name, history in metrics_history.items():
            for entry in history:
                rows.append({
                    'metric': metric_name,
                    'value': entry['value'],
                    'step': entry.get('step'),
                    'epoch': entry.get('epoch'),
                    'timestamp': entry.get('timestamp'),
                })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def _export_summary_to_csv(self, log_data: Dict, path: str):
        """导出会话摘要到CSV"""
        summary = {
            'session_id': log_data.get('session_id'),
            'task_name': log_data.get('task_name'),
            'start_time': log_data.get('start_time'),
            'end_time': log_data.get('end_time'),
            'status': log_data.get('status'),
            'total_logs': len(log_data.get('logs', [])),
        }

        # 添加最终指标
        final_metrics = log_data.get('final_metrics', {})
        for name, value in final_metrics.items():
            summary[f'final_{name}'] = value

        # 添加最佳指标
        best_metrics = log_data.get('best_metrics', {})
        for name, info in best_metrics.items():
            if isinstance(info, dict):
                summary[f'best_{name}'] = info.get('value')
            else:
                summary[f'best_{name}'] = info

        df = pd.DataFrame([summary])
        df.to_csv(path, index=False)


class HTMLExporter(LogExporter):
    """
    HTML导出器

    将日志导出为HTML报告。
    """

    def __init__(self, include_charts: bool = True):
        """
        初始化HTML导出器

        Args:
            include_charts: 是否包含图表
        """
        self.include_charts = include_charts

    def export(self, log_data: Dict, output_path: str) -> str:
        """导出为HTML"""
        if not output_path.endswith('.html'):
            output_path = f"{output_path}.html"

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        html_content = self._generate_html(log_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Exported to HTML: {output_path}")
        return output_path

    def _generate_html(self, log_data: Dict) -> str:
        """生成HTML内容"""
        session_id = log_data.get('session_id', 'Unknown')
        task_name = log_data.get('task_name', 'Training')
        status = log_data.get('status', 'Unknown')
        start_time = log_data.get('start_time', '')
        end_time = log_data.get('end_time', '')

        # 构建HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 4px; border-left: 4px solid #007bff; }}
        .summary-card h4 {{ margin: 0 0 5px 0; color: #666; font-size: 12px; text-transform: uppercase; }}
        .summary-card p {{ margin: 0; font-size: 18px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .status-completed {{ color: #28a745; }}
        .status-failed {{ color: #dc3545; }}
        .status-running {{ color: #ffc107; }}
        .metrics-table {{ margin-top: 20px; }}
        .log-entry {{ padding: 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; font-size: 12px; }}
        .log-INFO {{ background: #e7f3ff; }}
        .log-WARNING {{ background: #fff3cd; }}
        .log-ERROR {{ background: #f8d7da; }}
        .log-DEBUG {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Report</h1>

        <div class="summary">
            <div class="summary-card">
                <h4>Session ID</h4>
                <p>{session_id}</p>
            </div>
            <div class="summary-card">
                <h4>Task Name</h4>
                <p>{task_name}</p>
            </div>
            <div class="summary-card">
                <h4>Status</h4>
                <p class="status-{status.lower()}">{status}</p>
            </div>
            <div class="summary-card">
                <h4>Start Time</h4>
                <p>{start_time[:19] if start_time else 'N/A'}</p>
            </div>
            <div class="summary-card">
                <h4>End Time</h4>
                <p>{end_time[:19] if end_time else 'N/A'}</p>
            </div>
        </div>

        <h2>Final Metrics</h2>
        {self._generate_metrics_table(log_data.get('final_metrics', {}))}

        <h2>Best Metrics</h2>
        {self._generate_best_metrics_table(log_data.get('best_metrics', {}))}

        <h2>Configuration</h2>
        {self._generate_config_table(log_data.get('config', {}))}

        <h2>Training Logs</h2>
        {self._generate_logs_section(log_data.get('logs', []))}

    </div>
</body>
</html>
"""
        return html

    def _generate_metrics_table(self, metrics: Dict) -> str:
        """生成指标表格"""
        if not metrics:
            return "<p>No metrics available</p>"

        rows = "".join(f"<tr><td>{k}</td><td>{v:.6f if isinstance(v, float) else v}</td></tr>"
                       for k, v in metrics.items())
        return f"""
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th></tr>
            {rows}
        </table>
        """

    def _generate_best_metrics_table(self, metrics: Dict) -> str:
        """生成最佳指标表格"""
        if not metrics:
            return "<p>No best metrics available</p>"

        rows = []
        for k, v in metrics.items():
            if isinstance(v, dict):
                value = v.get('value', 'N/A')
                step = v.get('step', 'N/A')
                rows.append(f"<tr><td>{k}</td><td>{value:.6f if isinstance(value, float) else value}</td><td>{step}</td></tr>")
            else:
                rows.append(f"<tr><td>{k}</td><td>{v:.6f if isinstance(v, float) else v}</td><td>N/A</td></tr>")

        return f"""
        <table class="metrics-table">
            <tr><th>Metric</th><th>Best Value</th><th>Step</th></tr>
            {"".join(rows)}
        </table>
        """

    def _generate_config_table(self, config: Dict) -> str:
        """生成配置表格"""
        if not config:
            return "<p>No configuration available</p>"

        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in config.items())
        return f"""
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            {rows}
        </table>
        """

    def _generate_logs_section(self, logs: List[Dict]) -> str:
        """生成日志部分"""
        if not logs:
            return "<p>No logs available</p>"

        # 只显示最近100条
        recent_logs = logs[-100:]

        entries = []
        for log in recent_logs:
            level = log.get('level', 'INFO')
            timestamp = log.get('timestamp', '')[:19]
            message = log.get('message', '')
            step = log.get('step', '')
            step_str = f"[Step {step}]" if step else ""

            entries.append(f'<div class="log-entry log-{level}">{timestamp} [{level}] {step_str} {message}</div>')

        return "".join(entries)


class TensorBoardExporter(LogExporter):
    """
    TensorBoard导出器

    将日志导出为TensorBoard格式。
    """

    def __init__(self):
        """初始化TensorBoard导出器"""
        pass

    def export(self, log_data: Dict, output_path: str) -> str:
        """导出为TensorBoard格式"""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                logger.error("tensorboard or tensorboardX is required for TensorBoard export")
                raise ImportError("tensorboard or tensorboardX is required")

        os.makedirs(output_path, exist_ok=True)

        writer = SummaryWriter(log_dir=output_path)

        try:
            # 导出指标历史
            metrics_history = log_data.get('metrics_history', {})
            for metric_name, history in metrics_history.items():
                for entry in history:
                    step = entry.get('step', 0)
                    value = entry.get('value', 0)
                    writer.add_scalar(metric_name, value, step)

            # 导出超参数
            config = log_data.get('config', {})
            final_metrics = log_data.get('final_metrics', {})
            if config and final_metrics:
                writer.add_hparams(config, final_metrics)

        finally:
            writer.close()

        logger.info(f"Exported to TensorBoard: {output_path}")
        return output_path
