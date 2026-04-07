"""
对话摘要节点模块。

定义 LangGraph 状态图中负责对话历史摘要的节点。
当消息历史超过 token 预算或条数上限时，将旧消息压缩为摘要并物理删除，
确保长期对话不会超出 LLM 上下文窗口限制。
"""

from typing import Any

from ...skills.summary import summarize_and_trim
from ...state.state import AgentState


async def summarize_conversation(state: AgentState) -> dict[str, Any]:
    """对话历史摘要的图节点。

    当消息历史超过阈值时执行：
    1. 计算保留区边界（按 token 预算从后往前保留）
    2. 对保留区之前的旧消息生成增量摘要
    3. 通过 RemoveMessage 物理删除旧消息
    4. 同步调整分析指针 last_analyzed_index

    节点输入：
        - state.messages: 对话消息历史
        - state.summary: 已有的对话摘要（可能为空）
        - state.last_analyzed_index: 分析指针位置

    节点输出：
        - summary: 更新后的对话摘要
        - messages: RemoveMessage 列表（删除旧消息）
        - last_analyzed_index: 调整后的分析指针

    :param state: 当前 Agent 状态。
    :return: 状态更新字典。
    """
    return await summarize_and_trim(
        messages=state.messages,
        existing_summary=state.summary,
        last_analyzed_index=state.last_analyzed_index,
    )
