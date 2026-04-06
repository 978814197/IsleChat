"""
Agent 上下文定义模块。

定义 LangGraph 的运行时上下文 AgentContext，它在整个图执行期间
保持不变（只读），用于携带不属于状态流转但节点需要访问的外部信息。
"""

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Agent 运行时上下文。

    与 AgentState 不同，上下文在图执行期间不会被修改，
    适合存放用户身份标识、会话配置等元信息。

    使用方式：在构建图时通过 config 传入，节点通过
    ``runtime.context`` 访问。
    """

    # 用户唯一标识，用于关联长期记忆等持久化数据
    user_id: int = Field(..., description="用户唯一标识 ID")
