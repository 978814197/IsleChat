"""
Agent 包。

IsleChat 的核心 Agent 模块，提供长期陪伴型聊天 Agent 的完整实现。

主要组件：
- graph: LangGraph 状态图（Agent 的执行引擎）
- state: 状态和上下文定义
- models: LLM 实例和数据模型
- skills: 可扩展的技能模块（记忆、回复生成等）
"""

from .graph import build_graph
from .state import AgentContext, AgentState

__all__ = ["build_graph", "AgentState", "AgentContext"]
