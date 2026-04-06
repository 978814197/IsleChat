"""
记忆技能包。

提供用户长期记忆的分析、提取和持久化能力，是 Agent 实现
个性化陪伴的核心技能。

主要组件：
- MemoryService: 记忆服务门面，提供 analyze / save / load 接口
- MemoryRepository: 记忆持久化仓库
- memory_service: 全局记忆服务单例
"""

from .service import MemoryService, memory_service

__all__ = ["MemoryService", "memory_service"]
