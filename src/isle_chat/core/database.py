"""
数据库持久化管理模块。

基于 LangGraph 官方的 langgraph-checkpoint-postgres 提供：
- AsyncPostgresSaver: 短期记忆（对话历史检查点），自动管理对话状态的持久化与恢复
- AsyncPostgresStore: 长期记忆（用户画像等跨会话共享数据），基于 namespace + key-value 存储

底层使用 psycopg（异步模式）连接 PostgreSQL，由 LangGraph 内部管理连接池。

使用方式::

    from isle_chat.core.database import create_checkpointer, create_store

    # 短期记忆：对话历史检查点
    async with create_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

    # 长期记忆：用户画像存储
    async with create_store() as store:
        graph = build_graph(store=store)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from .settings import settings


@asynccontextmanager
async def create_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """创建异步 PostgreSQL 检查点保存器（短期记忆）。

    使用 LangGraph 官方的 AsyncPostgresSaver 管理对话状态检查点，
    自动在 PostgreSQL 中创建所需的表结构。

    通过 ``thread_id`` 配置项区分不同的对话线程，同一线程内的
    消息历史会自动持久化和恢复。

    :yields: 初始化完成的 AsyncPostgresSaver 实例。

    使用示例::

        async with create_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "user-1-thread-1"}}
            await graph.ainvoke(state, config=config)
    """
    async with AsyncPostgresSaver.from_conn_string(settings.db.dsn) as checkpointer:
        # 首次使用时自动创建检查点所需的表结构（幂等操作）
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def create_store() -> AsyncIterator[AsyncPostgresStore]:
    """创建异步 PostgreSQL 键值存储（长期记忆）。

    使用 LangGraph 官方的 AsyncPostgresStore 管理跨会话的长期记忆，
    数据以 namespace + key-value 的形式组织。

    存储结构约定::

        namespace = ("users", "<user_id>")
        key = "profile"
        value = {"name": "...", "city": "...", "profession": "...", "preferences": [...]}

    :yields: 初始化完成的 AsyncPostgresStore 实例。

    使用示例::

        async with create_store() as store:
            graph = build_graph(store=store)
            # 节点中通过 runtime.store 访问
    """
    async with AsyncPostgresStore.from_conn_string(settings.db.dsn) as store:
        # 首次使用时自动创建存储所需的表结构（幂等操作）
        await store.setup()
        yield store
