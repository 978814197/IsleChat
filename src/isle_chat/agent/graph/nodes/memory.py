"""
记忆相关节点模块。

定义 LangGraph 状态图中与长期记忆相关的节点，包括：
- should_extract_user_info: 分析对话是否包含需要记忆的用户信息
- save_user_info: 将提取的用户信息持久化到长期记忆中

这些节点都是"薄节点"，业务逻辑委托给 skills/memory 服务处理。
长期记忆通过 runtime.store（LangGraph Store）访问。
"""

from typing import Any

from langgraph.runtime import Runtime

from ...skills.memory import memory_service
from ...state.context import AgentContext
from ...state.state import AgentState


async def should_extract_user_info(state: AgentState) -> dict[str, Any]:
    """分析对话并判断是否需要提取用户信息的图节点。

    取出最近一轮对话（通常为一问一答），调用记忆分析服务判断
    对话中是否包含适合写入长期记忆的稳定用户信息（如姓名、
    城市、职业、偏好等）。

    节点输入：
        - state.latest_turn: 最近一轮对话消息

    节点输出：
        - should_extract_user_info: 是否需要提取用户信息
        - extracted_user_info: 提取出的用户信息（或 None）

    :param state: 当前 Agent 状态。
    :return: 状态更新字典，包含提取判断结果。
    """
    # 调用记忆分析服务，分析最近一轮对话
    result = await memory_service.analyze(state.latest_turn)

    return {
        "should_extract_user_info": result.should_extract,
        "extracted_user_info": result.user_info,
    }


async def save_user_info(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """将提取的用户信息保存到长期记忆中的图节点。

    当 should_extract_user_info 节点判断需要提取时，本节点负责
    将提取的用户信息通过记忆服务持久化到 LangGraph Store，并重置提取标志。

    节点输入：
        - state.extracted_user_info: 待保存的用户信息
        - runtime.context.user_id: 用户唯一标识（从上下文获取）
        - runtime.store: LangGraph Store 实例（长期记忆存储）

    节点输出：
        - should_extract_user_info: 重置为 False
        - extracted_user_info: 重置为 None

    :param state: 当前 Agent 状态。
    :param runtime: LangGraph 运行时，携带 AgentContext 上下文和 Store 实例。
    :return: 状态更新字典，重置提取相关标志。
    """
    # 将提取的用户信息持久化到 LangGraph Store（长期记忆）
    if state.extracted_user_info is not None and runtime.store is not None:
        await memory_service.save(
            store=runtime.store,
            user_id=runtime.context.user_id,
            user_info=state.extracted_user_info,
        )

    # 重置提取标志，表示本轮提取流程已完成
    return {
        "should_extract_user_info": False,
        "extracted_user_info": None,
    }
