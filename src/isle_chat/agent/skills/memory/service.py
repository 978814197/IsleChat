"""
记忆服务门面模块。

组合记忆分析器（analyzer）和记忆仓库（repository），
为上层的 graph 节点提供统一的记忆操作接口。

graph 节点只需调用 MemoryService 的方法，无需关心
分析和存储的内部实现细节。
"""

from langchain.messages import AnyMessage
from langgraph.store.base import BaseStore

from ...models.schemas import AgentProfile, AgentProfileExtractionResult, UserInfo, UserInfoExtractionResult
from .analyzer import analyze_agent_profile as _analyze_agent_profile
from .analyzer import analyze_conversation
from .repository import MemoryRepository


class MemoryService:
    """记忆服务。

    作为记忆技能的统一入口，封装了以下核心能力：
    - analyze: 分析对话是否包含需要记忆的用户信息
    - save: 将提取的用户信息持久化到 LangGraph Store
    - load: 从 LangGraph Store 加载用户的长期记忆

    所有涉及持久化的方法都需要传入 ``store`` 参数，
    该参数在 graph 节点中通过 ``runtime.store`` 获取。

    使用方式::

        memory_service = MemoryService()

        # 分析对话
        result = await memory_service.analyze(latest_turn)

        # 保存信息（store 来自 runtime.store）
        await memory_service.save(store, user_id=1, user_info=result.user_info)

        # 加载记忆
        user_info = await memory_service.load(store, user_id=1)
    """

    def __init__(self, repository: MemoryRepository | None = None) -> None:
        """初始化记忆服务。

        :param repository: 记忆仓库实例。若不传则使用默认的 LangGraph Store 实现。
        """
        self._repository = repository or MemoryRepository()

    async def analyze(self, latest_turn: list[AnyMessage]) -> UserInfoExtractionResult:
        """分析最近一轮对话，判断是否包含需要记忆的用户信息。

        将分析工作委托给 analyzer 模块，该模块使用分析模型（较小的 LLM）
        来判断对话中是否包含姓名、城市、职业、偏好等稳定信息。

        :param latest_turn: 最近一轮对话消息列表。
        :return: 提取结果，包含是否需要提取的标志和提取出的用户信息。
        """
        return await analyze_conversation(latest_turn)

    async def analyze_agent_profile(self, latest_turn: list[AnyMessage]) -> AgentProfileExtractionResult:
        """分析最近一轮对话，判断是否包含用户对 Agent 身份/行为的设置指令。

        将分析工作委托给 analyzer 模块，判断用户是否在设置 Agent 的
        名字、性格、称呼方式、说话风格或其他自定义指令。

        :param latest_turn: 最近一轮对话消息列表。
        :return: 提取结果，包含是否需要提取的标志和提取出的 Agent 配置。
        """
        return await _analyze_agent_profile(latest_turn)

    async def save(self, store: BaseStore, user_id: int, user_info: UserInfo) -> None:
        """将提取的用户信息保存到长期记忆中。

        使用 LangGraph Store 的 put 操作实现持久化。如果用户已有记忆记录，
        会自动合并更新（保留旧的非空字段，用新提取的非空字段覆盖，
        偏好列表去重合并）。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :param user_info: 待保存的用户信息。
        """
        await self._repository.save(store, user_id, user_info)

    async def load(self, store: BaseStore, user_id: int) -> UserInfo | None:
        """从长期记忆中加载用户信息。

        用于在对话开始时加载用户画像，注入到 system prompt 中
        实现个性化陪伴。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :return: 用户信息对象，若不存在则返回 None。
        """
        return await self._repository.load(store, user_id)

    async def load_agent_profile(self, store: BaseStore, user_id: int) -> AgentProfile | None:
        """从长期记忆中加载用户设置的 Agent 个性化配置。

        用于在对话开始时加载 Agent 的名字、性格、称呼方式等配置，
        注入到 system prompt 中影响 Agent 的回复行为。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :return: Agent 配置对象，若不存在则返回 None。
        """
        return await self._repository.load_agent_profile(store, user_id)

    async def save_agent_profile(
        self, store: BaseStore, user_id: int, agent_profile: AgentProfile
    ) -> None:
        """将用户设置的 Agent 个性化配置保存到长期记忆中。

        如果已有配置，会自动合并更新（新值非空时覆盖旧值）。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :param agent_profile: 待保存的 Agent 配置。
        """
        await self._repository.save_agent_profile(store, user_id, agent_profile)


# 全局记忆服务实例（单例）
# 在整个 Agent 生命周期内共享同一个记忆仓库
memory_service = MemoryService()
