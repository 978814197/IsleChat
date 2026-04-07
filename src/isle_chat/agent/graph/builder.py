"""
状态图构建模块。

负责构建和编译 LangGraph 状态图，定义 Agent 的整体执行流程。

当前图流程（三层记忆分析优化）::

    START
      │
      ▼
    load_memory                        ← 从 Store 加载用户画像 + Agent 配置
      │
      ▼
    generate_assistant_response        ← 生成 AI 回复（带记忆上下文）
      │
      ▼
    check_and_analyze                  ← 规则预筛 + 合并分析（一次 LLM 调用）
      │                                  若预筛未命中且未到兜底轮次 → 跳过分析
      │
      ├─(需要保存用户信息)→ save_user_info ──→ END
      ├─(需要保存Agent配置)→ save_agent_profile ──→ END
      └─(都不需要)→ END

优化效果：
- 闲聊对话：0 次分析 LLM 调用（规则预筛过滤）
- 有效对话：1 次分析 LLM 调用（合并分析，原来需要 2 次）
- 兜底机制：每 N 轮强制分析，防止遗漏
- 指针机制：避免重复分析已处理的消息

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
from .edges import route_after_analysis
from .nodes import (
    check_and_analyze,
    generate_assistant_response,
    load_memory,
    save_agent_profile,
    save_user_info,
)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """构建并编译 Agent 状态图。

    创建一个 LangGraph 状态图，注册所有节点和边，
    然后编译为可执行的图实例。

    图的执行流程（三层记忆分析优化）：
    1. 接收用户消息后，首先从 Store 加载用户画像和 Agent 配置
    2. 基于加载的记忆上下文生成 AI 回复
    3. 进入 check_and_analyze 节点：
       a. 规则预筛：关键词匹配判断是否可能包含有价值的信息
       b. 兜底判断：检查距上次分析是否已达 N 轮
       c. 若触发分析 → 一次 LLM 调用同时提取用户信息和 Agent 配置
       d. 若未触发 → 跳过分析，零 LLM 成本
    4. 根据分析结果条件路由到保存节点（可并行）

    :param checkpointer: 检查点保存器（短期记忆），用于持久化对话历史。
        传入后，同一 thread_id 的对话消息会自动保存和恢复。
    :param store: 键值存储（长期记忆），用于持久化用户画像等跨会话数据。
        传入后，节点可通过 runtime.store 访问。
    :return: 编译后的状态图，可通过 ainvoke() 执行。
    """
    graph = StateGraph(AgentState, AgentContext)

    # ── 注册节点 ──
    graph.add_node("load_memory", load_memory)
    graph.add_node("generate_assistant_response", generate_assistant_response)
    graph.add_node("check_and_analyze", check_and_analyze)
    graph.add_node("save_user_info", save_user_info)
    graph.add_node("save_agent_profile", save_agent_profile)

    # ── 定义边（执行流程）──

    # 入口 → 加载长期记忆（用户画像 + Agent 配置）
    graph.add_edge(START, "load_memory")

    # 加载记忆 → 生成回复（此时 state 中已有 memory_context 和 agent_profile）
    graph.add_edge("load_memory", "generate_assistant_response")

    # 生成回复 → 预筛判断 + 合并分析
    graph.add_edge("generate_assistant_response", "check_and_analyze")

    # 分析完成后 → 根据结果条件路由（可并行保存，或直接结束）
    graph.add_conditional_edges(
        "check_and_analyze",
        route_after_analysis,
        {
            "save_user_info": "save_user_info",
            "save_agent_profile": "save_agent_profile",
            END: END,
        },
    )

    # 保存完成 → 结束
    graph.add_edge("save_user_info", END)
    graph.add_edge("save_agent_profile", END)

    # 编译图，注入持久化组件
    return graph.compile(checkpointer=checkpointer, store=store)
