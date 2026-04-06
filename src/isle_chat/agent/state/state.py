"""
Agent 状态定义模块。

定义 LangGraph 状态图中流转的核心状态对象 AgentState，
它在图的每个节点之间传递，承载消息历史、记忆上下文和控制标志。
"""

from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field

from ..models.schemas import AgentProfile, UserInfo


class AgentState(BaseModel):
    """Agent 核心状态。

    在 LangGraph 状态图中流转的状态对象，包含：
    - messages: 完整的对话消息历史（通过 add_messages reducer 自动合并）
    - memory_context: 从长期记忆中加载的用户信息，用于个性化回复
    - should_extract_user_info: 控制标志，指示是否需要提取用户信息
    - extracted_user_info: 本轮提取出的用户信息（待保存）
    """

    # ── 对话消息 ──
    # add_messages reducer 会自动将新消息追加到列表末尾，而非覆盖
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="对话消息历史列表",
    )

    # ── 长期记忆上下文 ──
    # 在对话开始时从持久化存储加载，注入到 system prompt 中实现个性化
    memory_context: UserInfo | None = Field(
        default=None,
        description="从长期记忆加载的用户信息，用于个性化对话",
    )

    # ── Agent 个性化配置 ──
    # 用户设置的 Agent 身份信息（名字、性格、称呼等），注入到 system prompt 中
    agent_profile: AgentProfile | None = Field(
        default=None,
        description="用户设置的 Agent 个性化配置（名字、性格、称呼等）",
    )

    # ── 用户信息提取控制 ──
    should_extract_user_info: bool = Field(
        default=False,
        description="是否需要提取并保存用户信息",
    )
    extracted_user_info: UserInfo | None = Field(
        default=None,
        description="本轮对话中提取出的用户信息（待持久化）",
    )

    # ── Agent 配置提取控制 ──
    should_extract_agent_profile: bool = Field(
        default=False,
        description="是否需要提取并保存 Agent 个性化配置",
    )
    extracted_agent_profile: AgentProfile | None = Field(
        default=None,
        description="本轮对话中提取出的 Agent 配置（待持久化）",
    )

    @computed_field
    @property
    def last_message(self) -> AnyMessage:
        """获取最后一条消息。

        :return: 消息列表中的最后一条消息。
        :raises IndexError: 当消息列表为空时抛出。
        """
        return self.messages[-1]

    @computed_field
    @property
    def latest_turn(self) -> list[AnyMessage]:
        """获取最新一轮对话（最后两条消息，通常是一问一答）。

        如果消息不足两条，则返回当前已有的全部消息。

        :return: 最新一轮对话的消息列表。
        """
        return self.messages[-2:]
