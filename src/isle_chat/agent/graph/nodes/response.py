"""
回复生成节点模块。

定义 LangGraph 状态图中负责生成 Agent 回复的节点。
该节点是一个"薄节点"——只负责从状态中取出数据、调用技能服务、
将结果写回状态，不包含具体的业务逻辑。
"""

from typing import Any

from ...skills.response import generate_response
from ...state.state import AgentState


async def generate_assistant_response(state: AgentState) -> dict[str, Any]:
    """生成助手回复的图节点。

    从状态中取出对话历史和记忆上下文，调用回复生成技能
    生成个性化的 AI 回复，然后将新消息追加到状态中。

    节点输入：
        - state.messages: 对话消息历史
        - state.memory_context: 用户长期记忆（可能为 None）

    节点输出：
        - messages: 包含生成的 AI 回复消息的列表

    :param state: 当前 Agent 状态。
    :return: 状态更新字典，包含新生成的回复消息。
    """
    # 调用回复生成技能，传入对话历史和记忆上下文
    message = await generate_response(
        messages=state.messages,
        memory_context=state.memory_context,
    )

    # 返回新消息，LangGraph 的 add_messages reducer 会自动追加到列表末尾
    return {"messages": [message]}
