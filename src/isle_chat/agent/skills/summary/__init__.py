"""
对话摘要技能模块。

提供对话历史的滑动窗口摘要功能，防止消息列表无限增长导致
超出 LLM 上下文窗口限制。

核心能力：
- 双条件判断：token 容量为主 + 消息条数兜底
- 增量摘要：基于已有摘要做更新，不重复总结
- 保留区机制：摘要后保留最近的消息，确保近期上下文完整
"""

from .service import should_summarize, summarize_and_trim

__all__ = ["should_summarize", "summarize_and_trim"]
