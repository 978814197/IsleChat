"""
回复生成技能包。

提供 Agent 的对话回复生成能力，支持基于长期记忆的个性化回复。
"""

from .service import generate_response

__all__ = ["generate_response"]
