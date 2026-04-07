"""
IsleChat 应用入口模块。

提供命令行启动入口，构建 Agent 状态图并执行一轮示例对话。
使用 LangGraph 官方的持久化组件管理记忆：
- AsyncPostgresSaver（checkpointer）：短期记忆，自动保存/恢复对话历史
- AsyncPostgresStore（store）：长期记忆，跨会话共享的用户画像信息
"""

import asyncio
import platform

from langchain.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .agent import AgentContext, AgentState, build_graph
from .core.database import create_checkpointer, create_store


async def main() -> None:
    """应用主函数。

    生命周期：
        1. 创建 checkpointer（短期记忆）和 store（长期记忆）
        2. 构建 Agent 状态图，注入持久化组件
        3. 执行对话（通过 thread_id 管理对话线程）
        4. 退出时自动释放数据库连接
    """
    # 使用异步上下文管理器创建持久化组件，退出时自动清理连接
    async with create_checkpointer() as checkpointer, create_store() as store:
        # 构建 Agent 执行引擎，注入短期记忆和长期记忆
        app = build_graph(checkpointer=checkpointer, store=store)

        # 配置对话线程：thread_id 用于区分不同的对话会话，
        # 同一 thread_id 下的消息历史会自动持久化和恢复
        config = RunnableConfig(
            configurable={"thread_id": "user-1-main"}
        )
        while True:
            # 获取用户输入
            user_input = input("请输入你的问题：")
            if not user_input:
                continue

            # 执行一轮对话
            response = await app.ainvoke(
                AgentState(messages=[HumanMessage(user_input)]),
                config=config,
                context=AgentContext(user_id=1),
            )

            # 打印最后一条 AI 回复
            messages: list[AnyMessage] = response["messages"]
            messages[-1].pretty_print()


if __name__ == "__main__":
    if platform.system() == "Windows":
        # psycopg 在 Windows 上不兼容默认的 ProactorEventLoop，
        # 必须使用 SelectorEventLoop（基于 SelectSelector）才能正常运行异步模式
        import selectors

        asyncio.run(
            main(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    else:
        # 非 Windows 系统默认使用 SelectorEventLoop，无需特殊处理
        asyncio.run(main())
