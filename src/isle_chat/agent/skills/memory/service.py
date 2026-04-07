"""
记忆服务门面模块。

组合记忆分析器（analyzer）和记忆仓库（repository），
为上层的 graph 节点提供统一的记忆操作接口。

graph 节点只需调用 MemoryService 的方法，无需关心
分析和存储的内部实现细节。

三层分析优化：
1. should_trigger_analysis: 规则预筛 + 兜底计数判断
2. analyze: 合并分析（一次 LLM 调用同时提取用户信息和 Agent 配置）
3. save / save_agent_profile: 持久化提取结果
"""

from langchain.messages import AnyMessage
from langgraph.store.base import BaseStore

from ...models.schemas import AgentProfile, ConversationAnalysisResult, UserInfo
from .analyzer import analyze_conversation, get_last_user_message, should_trigger_analysis
from .repository import MemoryRepository


class MemoryService:
    """记忆服务。

    作为记忆技能的统一入口，封装了以下核心能力：
    - should_trigger: 规则预筛 + 兜底计数，判断是否需要触发 LLM 分析
    - analyze: 统一分析对话（一次 LLM 调用同时提取用户信息和 Agent 配置）
    - save / save_agent_profile: 将提取的信息持久化到 LangGraph Store
    - load / load_agent_profile: 从 LangGraph Store 加载长期记忆
    """

    def __init__(self, repository: MemoryRepository | None = None) -> None:
        """初始化记忆服务。

        :param repository: 记忆仓库实例。若不传则使用默认的 LangGraph Store 实现。
        """
        self._repository = repository or MemoryRepository()

    @staticmethod
    def should_trigger(
        messages: list[AnyMessage],
        turns_since_last_analysis: int,
    ) -> bool:
        """判断是否需要触发 LLM 分析。

        基于规则预筛和兜底计数器判断，零 LLM 成本。

        :param messages: 当前对话消息列表。
        :param turns_since_last_analysis: 距上次分析经过的轮次数。
        :return: 是否需要触发分析。
        """
        user_message = get_last_user_message(messages)
        return should_trigger_analysis(user_message, turns_since_last_analysis)

    @staticmethod
    async def analyze(messages: list[AnyMessage]) -> ConversationAnalysisResult:
        """统一分析对话，同时判断用户信息和 Agent 配置的提取需求。

        一次 LLM 调用同时完成用户信息提取和 Agent 配置提取，
        减少 API 调用次数和延迟。

        :param messages: 待分析的消息列表（从上次分析位置到当前的所有消息）。
        :return: 统一分析结果。
        """
        return await analyze_conversation(messages)

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

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :return: 用户信息对象，若不存在则返回 None。
        """
        return await self._repository.load(store, user_id)

    async def load_agent_profile(self, store: BaseStore, user_id: int) -> AgentProfile | None:
        """从长期记忆中加载用户设置的 Agent 个性化配置。

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
