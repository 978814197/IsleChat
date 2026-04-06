"""
记忆持久化仓库模块。

负责用户长期记忆在 LangGraph Store 中的存储和检索。
使用 LangGraph 官方的 BaseStore（AsyncPostgresStore）实现，
数据以 namespace + key-value 的形式组织。

存储结构::

    namespace = ("users", "<user_id>")
    key = "profile"
    value = {
        "name": "用户姓名",
        "city": "所在城市",
        "profession": "职业",
        "preferences": ["偏好1", "偏好2"]
    }
"""

from langgraph.store.base import BaseStore

from ...models.schemas import AgentProfile, UserInfo

# 存储中用于用户画像的固定 key
_PROFILE_KEY = "profile"
# 存储中用于 Agent 个性化配置的固定 key
_AGENT_PROFILE_KEY = "agent_profile"


class MemoryRepository:
    """记忆仓库（LangGraph Store 实现）。

    使用 LangGraph 官方的 BaseStore 接口访问 PostgreSQL，
    以 namespace（"users", user_id）为隔离单元，存储和检索用户长期记忆。

    合并策略：
        - 偏好列表（preferences）执行去重合并
        - 其他字段（name、city、profession）新值非空时覆盖旧值
    """

    def _build_namespace(self, user_id: int) -> tuple[str, str]:
        """构建用户记忆的 namespace。

        :param user_id: 用户唯一标识 ID。
        :return: 格式为 ("users", "<user_id>") 的 namespace 元组。
        """
        return ("users", str(user_id))

    async def load(self, store: BaseStore, user_id: int) -> UserInfo | None:
        """从 Store 中加载指定用户的记忆信息。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :return: 用户信息对象，若不存在则返回 None。
        """
        namespace = self._build_namespace(user_id)

        # 通过 namespace + key 获取用户画像
        item = await store.aget(namespace, _PROFILE_KEY)

        if item is None:
            return None

        # 将 Store 中的字典数据转换为 UserInfo 对象
        return UserInfo.model_validate(item.value)

    async def save(self, store: BaseStore, user_id: int, user_info: UserInfo) -> None:
        """保存或更新指定用户的记忆信息。

        合并策略：
        - 若用户记忆不存在，直接写入新记录
        - 若用户记忆已存在，执行智能合并：
          - name / city / profession: 新值非空时覆盖旧值
          - preferences: 合并去重

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :param user_info: 待保存的用户信息。
        """
        namespace = self._build_namespace(user_id)

        # 先加载已有记忆，用于合并更新
        existing = await store.aget(namespace, _PROFILE_KEY)

        if existing is not None:
            # 已有记忆，执行合并：新值非空时覆盖旧值，偏好列表去重合并
            old = existing.value
            merged = {
                "name": user_info.name or old.get("name"),
                "city": user_info.city or old.get("city"),
                "profession": user_info.profession or old.get("profession"),
                "preferences": list(dict.fromkeys(
                    old.get("preferences", []) + (user_info.preferences or [])
                )),
            }
        else:
            # 新用户，直接使用提取的信息
            merged = user_info.model_dump()

        # 写入 Store（put 操作是幂等的，key 相同则覆盖）
        await store.aput(namespace, _PROFILE_KEY, merged)

    async def load_agent_profile(self, store: BaseStore, user_id: int) -> AgentProfile | None:
        """从 Store 中加载指定用户设置的 Agent 个性化配置。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :return: Agent 配置对象，若不存在则返回 None。
        """
        namespace = self._build_namespace(user_id)
        item = await store.aget(namespace, _AGENT_PROFILE_KEY)

        if item is None:
            return None

        return AgentProfile.model_validate(item.value)

    async def save_agent_profile(
        self, store: BaseStore, user_id: int, agent_profile: AgentProfile
    ) -> None:
        """保存或更新指定用户设置的 Agent 个性化配置。

        合并策略：新值非空时覆盖旧值，保留旧的非空字段。

        :param store: LangGraph Store 实例（通过 runtime.store 获取）。
        :param user_id: 用户唯一标识 ID。
        :param agent_profile: 待保存的 Agent 配置。
        """
        namespace = self._build_namespace(user_id)

        # 先加载已有配置，用于合并更新
        existing = await store.aget(namespace, _AGENT_PROFILE_KEY)

        if existing is not None:
            # 已有配置，执行合并：新值非空时覆盖旧值
            old = existing.value
            merged = {
                "agent_name": agent_profile.agent_name or old.get("agent_name"),
                "user_nickname": agent_profile.user_nickname or old.get("user_nickname"),
                "personality": agent_profile.personality or old.get("personality"),
                "speaking_style": agent_profile.speaking_style or old.get("speaking_style"),
                "custom_instructions": agent_profile.custom_instructions or old.get("custom_instructions"),
            }
        else:
            merged = agent_profile.model_dump()

        await store.aput(namespace, _AGENT_PROFILE_KEY, merged)
