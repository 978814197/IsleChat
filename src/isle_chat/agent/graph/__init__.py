"""
状态图包。

导出图构建函数，供外部调用以创建 Agent 执行引擎。
"""

from .builder import build_graph

__all__ = ["build_graph"]
