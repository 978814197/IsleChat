"""
图边（路由）定义模块。

定义 LangGraph 状态图中的条件边路由函数。
这些函数根据当前状态决定下一个要执行的节点。

路由逻辑：
- load_memory 后判断是否需要摘要压缩对话历史
- check_and_analyze 节点执行完毕后，根据分析结果并行路由到保存节点
"""

from typing import Union

from langgraph.constants import END, Send

from ..skills.summary import should_summarize
from ..state.state import AgentState


def route_after_load_memory(state: AgentState) -> str:
    """加载记忆后的路由函数。

    判断当前消息历史是否需要摘要压缩：
    - 需要摘要 → summarize_conversation
    - 不需要 → generate_assistant_response

    使用双条件判断：token 容量为主 + 消息条数兜底。

    :param state: 当前 Agent 状态。
    :return: 下一步要执行的节点名称。
    """
    if should_summarize(state.messages):
        return "summarize_conversation"
    return "generate_assistant_response"


def route_after_analysis(state: AgentState) -> Union[str, list[Send]]:
    """分析完成后的路由函数。

    在 check_and_analyze 节点执行完毕后，根据分析结果决定
    需要执行哪些保存节点：
    - 需要保存用户信息和 Agent 配置 → 通过 Send 并行执行两个保存节点
    - 只需要保存其中一个 → 直接路由到对应节点
    - 都不需要 → 直接结束

    :param state: 当前 Agent 状态。
    :return: 单个节点名称（str）或 Send 列表（并行 fan-out）。
    """
    targets: list[Send] = []

    if state.should_extract_user_info and state.extracted_user_info is not None:
        targets.append(Send("save_user_info", state))

    if state.should_extract_agent_profile and state.extracted_agent_profile is not None:
        targets.append(Send("save_agent_profile", state))

    if not targets:
        return END
    if len(targets) == 1:
        # 单个目标时直接返回节点名，无需 Send
        return targets[0].node
    # 多个目标时返回 Send 列表，实现并行 fan-out
    return targets
