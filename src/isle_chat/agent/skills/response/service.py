"""
回复生成服务模块。

负责生成 Agent 的对话回复，包括：
1. 构建包含用户记忆上下文的 system prompt
2. 调用主模型生成回复

system prompt 会根据用户的长期记忆动态调整，
使 Agent 能够记住用户的姓名、偏好等信息，提供个性化陪伴。
"""

from langchain.messages import AnyMessage, SystemMessage
from langchain_core.messages import AIMessage

from ...models.llm import get_primary_llm
from ...models.schemas import AgentProfile, UserInfo

# Agent 默认基础人设 prompt（当用户未设置 Agent 配置时使用）
_DEFAULT_SYSTEM_PROMPT = (
    "你是一个温暖、友善的长期陪伴型聊天助手。"
    "你会记住用户告诉你的信息，并在后续对话中自然地引用这些信息。"
    "保持轻松愉快的对话风格，像朋友一样陪伴用户。"
)

# Agent 个性化配置模板，当用户设置了 Agent 配置时用于构建人设 prompt
_AGENT_PROFILE_TEMPLATE = (
    "你是一个长期陪伴型聊天助手。"
    "你会记住用户告诉你的信息，并在后续对话中自然地引用这些信息。"
    "\n\n以下是你的身份设定，请始终以这个身份与用户交流：\n"
    "{agent_details}"
)

# 用户记忆上下文模板，当有记忆信息时拼接到 system prompt 中
_MEMORY_CONTEXT_TEMPLATE = (
    "\n\n以下是你已知的关于这位用户的信息，请在对话中自然地运用：\n"
    "{memory_details}"
)

# 首次引导用户设置个人信息和 Agent 配置的 prompt 片段
_SETUP_PROMPT_HINT = (
    "\n\n【重要】这是你与用户的初次交流，用户还没有设置任何个人信息和你的身份配置。"
    "请在自然回复的同时，友好地引导用户告诉你以下信息："
    "\n关于用户自己："
    "\n- 名字或称呼（例如'小明'、'阿花'）"
    "\n- 所在城市"
    "\n- 职业"
    "\n- 兴趣爱好或偏好"
    "\n关于你（Agent）的设定："
    "\n- 给你起一个名字（例如'小岛'）"
    "\n- 希望你怎么称呼用户（例如'主人'、'小屿'）"
    "\n- 你的性格（例如'温柔体贴'、'活泼可爱'）"
    "\n- 你的说话风格（例如'喜欢用颜文字'、'语气俏皮'）"
    "\n- 其他额外设定或指令"
    "\n\n注意：语气要自然轻松，像朋友一样建议，不要像表单一样逐条列举。"
    "可以在聊天过程中自然地引出这些话题，不需要一次性全部问完。"
    "只需要提一次，不要反复催促。"
)


def _build_system_prompt(
        memory_context: UserInfo | None = None,
        agent_profile: AgentProfile | None = None,
        should_prompt_setup: bool = False,
) -> SystemMessage:
    """构建包含 Agent 人设和用户记忆上下文的 system prompt。

    prompt 构建顺序：
    1. Agent 人设（若用户设置了 Agent 配置则使用自定义人设，否则使用默认人设）
    2. 用户记忆上下文（若存在）
    3. 首次引导提示（若需要引导用户设置）

    :param memory_context: 从长期记忆加载的用户信息，可能为 None。
    :param agent_profile: 用户设置的 Agent 个性化配置，可能为 None。
    :param should_prompt_setup: 是否需要引导用户设置个人信息和 Agent 配置。
    :return: 构建好的 SystemMessage。
    """
    # ── 第一部分：Agent 人设 ──
    if agent_profile is not None:
        # 用户设置了 Agent 配置，构建自定义人设
        agent_details = []
        if agent_profile.agent_name:
            agent_details.append(f"- 你的名字是：{agent_profile.agent_name}")
        if agent_profile.user_nickname:
            agent_details.append(f"- 你称呼用户为：{agent_profile.user_nickname}")
        if agent_profile.personality:
            agent_details.append(f"- 你的性格：{agent_profile.personality}")
        if agent_profile.speaking_style:
            agent_details.append(f"- 你的说话风格：{agent_profile.speaking_style}")
        if agent_profile.custom_instructions:
            agent_details.append(f"- 额外设定：{agent_profile.custom_instructions}")

        if agent_details:
            prompt = _AGENT_PROFILE_TEMPLATE.format(
                agent_details="\n".join(agent_details),
            )
        else:
            # Agent 配置存在但所有字段都为空，回退到默认人设
            prompt = _DEFAULT_SYSTEM_PROMPT
    else:
        # 未设置 Agent 配置，使用默认人设
        prompt = _DEFAULT_SYSTEM_PROMPT

    # ── 第二部分：用户记忆上下文 ──
    if memory_context is not None:
        # 将用户信息格式化为可读的文本片段
        details = []
        if memory_context.name:
            details.append(f"- 姓名/称呼：{memory_context.name}")
        if memory_context.city:
            details.append(f"- 所在城市：{memory_context.city}")
        if memory_context.profession:
            details.append(f"- 职业：{memory_context.profession}")
        if memory_context.preferences:
            details.append(f"- 偏好/兴趣：{'、'.join(memory_context.preferences)}")

        # 只有当确实有有效信息时才追加记忆上下文
        if details:
            prompt += _MEMORY_CONTEXT_TEMPLATE.format(
                memory_details="\n".join(details),
            )

    # ── 第三部分：首次引导提示 ──
    if should_prompt_setup:
        prompt += _SETUP_PROMPT_HINT

    return SystemMessage(content=prompt)


async def generate_response(
        messages: list[AnyMessage],
        memory_context: UserInfo | None = None,
        agent_profile: AgentProfile | None = None,
        should_prompt_setup: bool = False,
) -> AIMessage:
    """生成 Agent 的对话回复。

    将用户的消息历史连同 system prompt（包含 Agent 人设和记忆上下文）
    一起发送给主模型，生成个性化的回复。

    :param messages: 完整的对话消息历史列表。
    :param memory_context: 从长期记忆加载的用户信息，用于个性化回复。
    :param agent_profile: 用户设置的 Agent 个性化配置，影响 Agent 的人设和行为。
    :param should_prompt_setup: 是否需要引导用户设置个人信息和 Agent 配置。
    :return: 模型生成的 AI 回复消息。
    """
    # 构建带有 Agent 人设、记忆上下文和引导提示的 system prompt
    system_message = _build_system_prompt(memory_context, agent_profile, should_prompt_setup)

    # 将 system prompt 放在消息列表最前面，然后拼接对话历史
    full_messages = [system_message] + messages

    # 调用主模型生成回复
    primary_llm = get_primary_llm()
    return await primary_llm.ainvoke(
        full_messages,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )
