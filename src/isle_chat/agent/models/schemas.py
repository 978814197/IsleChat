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


class AgentProfile(BaseModel):
    """Agent 个性化配置。

    由用户设置的 Agent 身份和行为信息，存储在长期记忆中。
    这些信息决定了 Agent 的人设、称呼方式和交互风格，
    会被注入到 system prompt 中影响 Agent 的回复行为。

    示例：
        用户可以设置 Agent 叫"小岛"，称呼用户为"主人"，
        性格设定为"活泼可爱、喜欢撒娇"。
    """

    agent_name: str | None = Field(default=None, description="Agent 的名字，例如'小岛'")
    user_nickname: str | None = Field(default=None, description="Agent 对用户的称呼，例如'主人'、'小屿'")
    personality: str | None = Field(default=None, description="Agent 的性格描述，例如'温柔体贴、善解人意'")
    speaking_style: str | None = Field(default=None, description="Agent 的说话风格，例如'喜欢用颜文字、语气活泼'")
    custom_instructions: str | None = Field(default=None, description="用户自定义的额外指令或设定")


class UserInfoExtractionResult(BaseModel):
    """用户信息提取结果。

    由分析模型返回的结构化结果，用于判断最近一轮对话中是否
    包含适合写入长期记忆的稳定用户信息。
    """

    should_extract: bool = Field(..., description="是否需要提取并保存用户信息")
    reason: str = Field(default="", description="判断原因（便于调试和日志记录）")
    user_info: UserInfo | None = Field(default=None, description="提取出的用户信息，仅在 should_extract=True 时有值")


class AgentProfileExtractionResult(BaseModel):
    """Agent 配置提取结果。

    由分析模型返回的结构化结果，用于判断最近一轮对话中是否
    包含用户对 Agent 身份/行为的设置指令（如改名、设定性格等）。
    """

    should_extract: bool = Field(..., description="是否需要提取并保存 Agent 配置")
    reason: str = Field(default="", description="判断原因（便于调试和日志记录）")
    agent_profile: AgentProfile | None = Field(default=None, description="提取出的 Agent 配置，仅在 should_extract=True 时有值")


class ConversationAnalysisResult(BaseModel):
    """统一的对话分析结果。

    将用户信息提取和 Agent 配置提取合并为一次 LLM 调用的结构化输出，
    同时判断对话中是否包含需要记忆的用户信息和 Agent 设置指令。
    """

    user_info_extraction: UserInfoExtractionResult = Field(
        ..., description="用户信息提取结果",
    )
    agent_profile_extraction: AgentProfileExtractionResult = Field(
        ..., description="Agent 配置提取结果",
    )
