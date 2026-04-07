"""
记忆相关节点模块。

定义 LangGraph 状态图中与长期记忆相关的节点，包括：
- check_and_analyze: 预筛判断 + 合并分析（规则预筛命中或兜底触发时调用 LLM）
- save_user_info: 将提取的用户信息持久化到长期记忆中
- save_agent_profile: 将提取的 Agent 配置持久化到长期记忆中

三层记忆分析优化：
1. 规则预筛：零成本关键词匹配，快速跳过闲聊
2. 合并分析：一次 LLM 调用同时提取用户信息和 Agent 配置
3. 兜底分析：每 N 轮强制分析一次，防止规则预筛遗漏

使用指针机制（last_analyzed_index）避免重复分析已处理的消息。
"""

from typing import Any

from langgraph.runtime import Runtime

from ...skills.memory import memory_service
from ...state.context import AgentContext
from ...state.state import AgentState


async def check_and_analyze(state: AgentState) -> dict[str, Any]:
    """预筛判断 + 合并分析的图节点。

    每轮对话进入此节点时：
    1. 轮次计数器 +1
    2. 规则预筛：检查用户最新消息是否命中关键词
    3. 兜底判断：检查轮次计数器是否达到阈值
    4. 若预筛命中或兜底触发 → 调用 LLM 合并分析（只分析指针之后的消息）
    5. 若都未命中 → 跳过分析，零 LLM 成本

    节点输入：
        - state.messages: 对话消息历史
        - state.turns_since_last_analysis: 距上次分析的轮次数
        - state.last_analyzed_index: 上次分析覆盖到的消息位置

    节点输出：
        - turns_since_last_analysis: 更新后的轮次计数器
        - analysis_result: 统一分析结果（或 None）
        - should_extract_user_info: 是否需要保存用户信息
        - extracted_user_info: 提取出的用户信息
        - should_extract_agent_profile: 是否需要保存 Agent 配置
        - extracted_agent_profile: 提取出的 Agent 配置
        - last_analyzed_index: 更新后的分析指针

    :param state: 当前 Agent 状态。
    :return: 状态更新字典。
    """
    # 轮次计数器 +1
    turns = state.turns_since_last_analysis + 1

    # 判断是否需要触发 LLM 分析（规则预筛 + 兜底计数）
    should_analyze = memory_service.should_trigger(
        messages=state.messages,
        turns_since_last_analysis=turns,
    )

    if not should_analyze:
        # 未触发分析，只更新计数器，跳过 LLM 调用
        return {
            "turns_since_last_analysis": turns,
            "analysis_result": None,
            "should_extract_user_info": False,
            "extracted_user_info": None,
            "should_extract_agent_profile": False,
            "extracted_agent_profile": None,
        }

    # 触发分析：只分析指针之后的未分析消息
    unanalyzed_messages = state.messages[state.last_analyzed_index:]

    # 一次 LLM 调用同时提取用户信息和 Agent 配置
    result = await memory_service.analyze(unanalyzed_messages)

    # 从统一结果中提取各部分
    user_extraction = result.user_info_extraction
    profile_extraction = result.agent_profile_extraction

    return {
        # 重置计数器，更新分析指针
        "turns_since_last_analysis": 0,
        "last_analyzed_index": len(state.messages),
        # 统一分析结果
        "analysis_result": result,
        # 用户信息提取结果
        "should_extract_user_info": user_extraction.should_extract,
        "extracted_user_info": user_extraction.user_info,
        # Agent 配置提取结果
        "should_extract_agent_profile": profile_extraction.should_extract,
        "extracted_agent_profile": profile_extraction.agent_profile,
    }


async def save_user_info(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """将提取的用户信息保存到长期记忆中的图节点。

    当 check_and_analyze 节点判断需要提取用户信息时，本节点负责
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
    if state.extracted_user_info is not None and runtime.store is not None:
        await memory_service.save(
            store=runtime.store,
            user_id=runtime.context.user_id,
            user_info=state.extracted_user_info,
        )

    return {
        "should_extract_user_info": False,
        "extracted_user_info": None,
    }


async def save_agent_profile(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """将提取的 Agent 配置保存到长期记忆中的图节点。

    当 check_and_analyze 节点判断需要提取 Agent 配置时，本节点负责
    将提取的 Agent 配置通过记忆服务持久化到 LangGraph Store，并重置提取标志。

    节点输入：
        - state.extracted_agent_profile: 待保存的 Agent 配置
        - runtime.context.user_id: 用户唯一标识（从上下文获取）
        - runtime.store: LangGraph Store 实例（长期记忆存储）

    节点输出：
        - should_extract_agent_profile: 重置为 False
        - extracted_agent_profile: 重置为 None

    :param state: 当前 Agent 状态。
    :param runtime: LangGraph 运行时，携带 AgentContext 上下文和 Store 实例。
    :return: 状态更新字典，重置提取相关标志。
    """
    if state.extracted_agent_profile is not None and runtime.store is not None:
        await memory_service.save_agent_profile(
            store=runtime.store,
            user_id=runtime.context.user_id,
            agent_profile=state.extracted_agent_profile,
        )

    return {
        "should_extract_agent_profile": False,
        "extracted_agent_profile": None,
    }
