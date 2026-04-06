"""
状态图构建模块。

负责构建和编译 LangGraph 状态图，定义 Agent 的整体执行流程。

当前图流程::

    START
      │
      ▼
    generate_assistant_response    ← 生成 AI 回复（带记忆上下文）
      │
      ▼
    should_extract_user_info       ← 分析对话是否包含用户信息
      │
      ├──（需要提取）──▶ save_user_info ──▶ END
      │
      └──（不需要）──▶ END

持久化支持：
- checkpointer: 短期记忆，自动保存/恢复对话历史（通过 thread_id 区分线程）
- store: 长期记忆，跨会话共享的用户画像信息（节点通过 runtime.store 访问）
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from ..state.context import AgentContext
from ..state.state import AgentState
from .edges import route_after_user_info_extraction
from .nodes import (
    generate_assistant_response,
    save_user_info,
    should_extract_user_info,
)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """构建并编译 Agent 状态图。

    创建一个 LangGraph 状态图，注册所有节点和边，
    然后编译为可执行的图实例。

    图的执行流程：
    1. 接收用户消息后，首先生成 AI 回复
    2. 然后分析本轮对话是否包含需要记忆的用户信息
    3. 如果有，保存到长期记忆中；如果没有，直接结束

    :param checkpointer: 检查点保存器（短期记忆），用于持久化对话历史。
        传入后，同一 thread_id 的对话消息会自动保存和恢复。
    :param store: 键值存储（长期记忆），用于持久化用户画像等跨会话数据。
        传入后，节点可通过 runtime.store 访问。
    :return: 编译后的状态图，可通过 ainvoke() 执行。
    """
    graph = StateGraph(AgentState, AgentContext)

    # ── 注册节点 ──
    graph.add_node("generate_assistant_response", generate_assistant_response)
    graph.add_node("should_extract_user_info", should_extract_user_info)
    graph.add_node("save_user_info", save_user_info)

    # ── 定义边（执行流程）──
    # 入口 → 生成回复
    graph.add_edge(START, "generate_assistant_response")
    # 生成回复 → 分析用户信息
    graph.add_edge("generate_assistant_response", "should_extract_user_info")
    # 分析完成后，根据结果条件路由
    graph.add_conditional_edges(
        "should_extract_user_info",
        route_after_user_info_extraction,
        {
            "save_user_info": "save_user_info",
            END: END,
        },
    )
    # 保存完成 → 结束
    graph.add_edge("save_user_info", END)

    # 编译图，注入持久化组件
    return graph.compile(checkpointer=checkpointer, store=store)
