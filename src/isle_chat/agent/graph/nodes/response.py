"""
回复生成节点模块。

定义 LangGraph 状态图中负责生成 Agent 回复的节点。
该节点是一个"薄节点"——只负责从状态中取出数据、调用技能服务、
将结果写回状态，不包含具体的业务逻辑。
"""

from typing import Any

from langgraph.runtime import Runtime

from ...skills.memory import memory_service
from ...skills.response import generate_response
from ...state.context import AgentContext
from ...state.state import AgentState


async def load_memory(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """从长期记忆中加载用户信息和 Agent 配置的图节点。

    在生成回复之前执行，从 LangGraph Store 中读取当前用户的
    长期记忆（用户画像）和 Agent 个性化配置，写入状态供后续节点使用。

    节点输入：
        - runtime.context.user_id: 用户唯一标识
        - runtime.store: LangGraph Store 实例（长期记忆存储）

    节点输出：
        - memory_context: 用户长期记忆信息（或 None）
        - agent_profile: Agent 个性化配置（或 None）

    :param state: 当前 Agent 状态。
    :param runtime: LangGraph 运行时，携带 AgentContext 上下文和 Store 实例。
    :return: 状态更新字典，包含加载的记忆信息。
    """
    memory_context = None
    agent_profile = None

    if runtime.store is not None:
        user_id = runtime.context.user_id

        # 从 Store 加载用户画像（姓名、城市、职业、偏好等）
        memory_context = await memory_service.load(runtime.store, user_id)

        # 从 Store 加载 Agent 个性化配置（名字、性格、称呼等）
        agent_profile = await memory_service.load_agent_profile(runtime.store, user_id)

    return {
        "memory_context": memory_context,
        "agent_profile": agent_profile,
    }


async def generate_assistant_response(state: AgentState) -> dict[str, Any]:
    """生成助手回复的图节点。

    从状态中取出对话历史和记忆上下文，调用回复生成技能
    生成个性化的 AI 回复，然后将新消息追加到状态中。

    节点输入：
        - state.messages: 对话消息历史
        - state.memory_context: 用户长期记忆（由 load_memory 节点加载）
        - state.agent_profile: Agent 个性化配置（由 load_memory 节点加载）

    节点输出：
        - messages: 包含生成的 AI 回复消息的列表

    :param state: 当前 Agent 状态。
    :return: 状态更新字典，包含新生成的回复消息。
    """
    # 调用回复生成技能，传入对话历史、记忆上下文和 Agent 个性化配置
    message = await generate_response(
        messages=state.messages,
        memory_context=state.memory_context,
        agent_profile=state.agent_profile,
    )

    # 返回新消息，LangGraph 的 add_messages reducer 会自动追加到列表末尾
    return {"messages": [message]}
