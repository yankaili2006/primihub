"""
Python Script Processing Base Classes
Python脚本处理基础类

提供Python脚本执行和验证功能。
"""

import logging
import sys
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScriptContext:
    """
    脚本执行上下文

    提供脚本执行时可用的变量和函数。
    """

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 labels: Optional[pd.Series] = None,
                 params: Optional[Dict] = None):
        """
        初始化脚本上下文

        Args:
            data: 输入数据
            labels: 标签数据
            params: 自定义参数
        """
        self.data = data
        self.labels = labels
        self.params = params or {}
        self.result = None
        self.output = {}
        self._logs = []

    def log(self, message: str):
        """记录日志"""
        self._logs.append(message)
        logger.info(f"Script: {message}")

    def set_result(self, result: Any):
        """设置返回结果"""
        self.result = result

    def set_output(self, key: str, value: Any):
        """设置输出变量"""
        self.output[key] = value

    def get_logs(self) -> List[str]:
        """获取日志"""
        return self._logs


class ScriptValidator:
    """
    脚本验证器

    验证脚本的安全性和语法正确性。
    """

    # 危险的导入和函数
    DANGEROUS_IMPORTS = [
        'os.system', 'subprocess', 'shutil.rmtree',
        '__import__', 'eval', 'exec', 'compile',
        'open',  # 限制文件操作
    ]

    DANGEROUS_PATTERNS = [
        'os.system', 'subprocess.', 'shutil.rmtree',
        '__import__', 'globals()', 'locals()',
        'setattr(', 'delattr(', 'getattr(',
    ]

    def __init__(self, allow_file_ops: bool = False,
                 allow_system_calls: bool = False):
        """
        初始化验证器

        Args:
            allow_file_ops: 是否允许文件操作
            allow_system_calls: 是否允许系统调用
        """
        self.allow_file_ops = allow_file_ops
        self.allow_system_calls = allow_system_calls

    def validate(self, script: str) -> Dict[str, Any]:
        """
        验证脚本

        Args:
            script: Python脚本代码

        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # 语法检查
        try:
            compile(script, '<script>', 'exec')
        except SyntaxError as e:
            result["valid"] = False
            result["errors"].append(f"Syntax error: {e}")
            return result

        # 安全检查
        if not self.allow_system_calls:
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in script:
                    result["warnings"].append(f"Potentially dangerous pattern: {pattern}")

        # 文件操作检查
        if not self.allow_file_ops:
            if 'open(' in script and 'context.data' not in script:
                result["warnings"].append("File operations may be restricted")

        return result


class ScriptRunner:
    """
    脚本执行器

    安全执行Python脚本。
    """

    def __init__(self, timeout: int = 60,
                 max_memory_mb: int = 1024):
        """
        初始化脚本执行器

        Args:
            timeout: 执行超时时间（秒）
            max_memory_mb: 最大内存限制（MB）
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def run(self, script: str, context: ScriptContext) -> Dict[str, Any]:
        """
        执行脚本

        Args:
            script: Python脚本代码
            context: 执行上下文

        Returns:
            执行结果
        """
        logger.info("Running Python script")

        result = {
            "success": False,
            "result": None,
            "output": {},
            "logs": [],
            "stdout": "",
            "stderr": "",
            "error": None,
        }

        # 捕获stdout和stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # 创建安全的执行环境
            safe_globals = self._create_safe_globals(context)

            # 执行脚本
            exec(script, safe_globals)

            result["success"] = True
            result["result"] = context.result
            result["output"] = context.output
            result["logs"] = context.get_logs()

        except Exception as e:
            result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            logger.error(f"Script execution failed: {e}")

        finally:
            result["stdout"] = sys.stdout.getvalue()
            result["stderr"] = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result

    def _create_safe_globals(self, context: ScriptContext) -> Dict[str, Any]:
        """创建安全的全局变量环境"""
        return {
            # 基础类型
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
            },
            # 数据处理库
            'np': np,
            'numpy': np,
            'pd': pd,
            'pandas': pd,
            # 上下文
            'context': context,
            'data': context.data,
            'labels': context.labels,
            'params': context.params,
        }
