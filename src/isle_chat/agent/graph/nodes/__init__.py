"""
图节点包。

导出所有可在状态图中注册的节点函数。
每个节点函数都是"薄节点"，只负责状态读写和技能服务调用。
"""

from .memory import (
    save_agent_profile,
    save_user_info,
    should_extract_agent_profile,
    should_extract_user_info,
)
from .response import generate_assistant_response, load_memory

__all__ = [
    "load_memory",
    "generate_assistant_response",
    "should_extract_user_info",
    "save_user_info",
    "should_extract_agent_profile",
    "save_agent_profile",
]
