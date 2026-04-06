"""
数据模型定义模块。

定义 Agent 中共享的 Pydantic 数据模型，包括用户信息结构
和 LLM 提取结果的结构化输出格式。
"""

from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """用户基本信息。

    用于长期记忆中存储的用户稳定信息，包括姓名、所在城市、
    职业、长期偏好等。这些信息会被注入到对话的 system prompt 中，
    帮助 Agent 提供个性化的陪伴体验。
    """

    name: str | None = Field(default=None, description="用户姓名或称呼")
    city: str | None = Field(default=None, description="用户所在城市")
    profession: str | None = Field(default=None, description="用户职业")
    preferences: list[str] = Field(default_factory=list, description="用户的长期偏好列表")


class UserInfoExtractionResult(BaseModel):
    """用户信息提取结果。

    由分析模型返回的结构化结果，用于判断最近一轮对话中是否
    包含适合写入长期记忆的稳定用户信息。
    """

    should_extract: bool = Field(..., description="是否需要提取并保存用户信息")
    reason: str = Field(default="", description="判断原因（便于调试和日志记录）")
    user_info: UserInfo | None = Field(default=None, description="提取出的用户信息，仅在 should_extract=True 时有值")
