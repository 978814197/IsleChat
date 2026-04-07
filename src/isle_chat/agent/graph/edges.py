"""
图边（路由）定义模块。

定义 LangGraph 状态图中的条件边路由函数。
这些函数根据当前状态决定下一个要执行的节点。

三层记忆分析优化后的路由逻辑：
- check_and_analyze 节点执行完毕后，根据分析结果并行路由到保存节点
- 若需要保存用户信息 → save_user_info
- 若需要保存 Agent 配置 → save_agent_profile
- 若都不需要 → 直接结束
"""

from langgraph.constants import END

from ..state.state import AgentState


def route_after_analysis(state: AgentState) -> list[str]:
    """分析完成后的路由函数。

    在 check_and_analyze 节点执行完毕后，根据分析结果决定
    需要并行执行哪些保存节点：
    - 需要保存用户信息 → 加入 save_user_info
    - 需要保存 Agent 配置 → 加入 save_agent_profile
    - 都不需要 → 直接结束

    使用 Send 列表实现并行 fan-out。

    :param state: 当前 Agent 状态。
    :return: 下一步要执行的节点名称列表。
    """
    targets: list[str] = []

    if state.should_extract_user_info and state.extracted_user_info is not None:
        targets.append("save_user_info")

    if state.should_extract_agent_profile and state.extracted_agent_profile is not None:
        targets.append("save_agent_profile")

    return targets if targets else [END]
