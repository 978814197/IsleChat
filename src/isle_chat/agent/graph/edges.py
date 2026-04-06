"""
图边（路由）定义模块。

定义 LangGraph 状态图中的条件边路由函数。
这些函数根据当前状态决定下一个要执行的节点。
"""

from langgraph.constants import END

from ..state.state import AgentState


def route_after_user_info_extraction(state: AgentState) -> str:
    """用户信息提取后的路由函数。

    在 should_extract_user_info 节点执行完毕后，根据提取结果
    决定下一步流向：
    - 如果需要提取且成功提取了用户信息 → 进入 save_user_info 节点保存
    - 否则 → 直接结束，跳过保存步骤

    :param state: 当前 Agent 状态。
    :return: 下一个节点的名称，"save_user_info" 或 END。
    """
    if state.should_extract_user_info and state.extracted_user_info is not None:
        return "save_user_info"
    return END
