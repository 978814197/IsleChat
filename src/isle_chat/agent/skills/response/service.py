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
from ...models.schemas import UserInfo


# Agent 基础人设 prompt
_BASE_SYSTEM_PROMPT = (
    "你是一个温暖、友善的长期陪伴型聊天助手。"
    "你会记住用户告诉你的信息，并在后续对话中自然地引用这些信息。"
    "保持轻松愉快的对话风格，像朋友一样陪伴用户。"
)

# 用户记忆上下文模板，当有记忆信息时拼接到 system prompt 中
_MEMORY_CONTEXT_TEMPLATE = (
    "\n\n以下是你已知的关于这位用户的信息，请在对话中自然地运用：\n"
    "{memory_details}"
)


def _build_system_prompt(memory_context: UserInfo | None) -> SystemMessage:
    """构建包含用户记忆上下文的 system prompt。

    如果存在用户的长期记忆信息，会将其格式化后追加到基础人设 prompt 中，
    使 Agent 能够个性化地回复用户。

    :param memory_context: 从长期记忆加载的用户信息，可能为 None。
    :return: 构建好的 SystemMessage。
    """
    prompt = _BASE_SYSTEM_PROMPT

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

    return SystemMessage(content=prompt)


async def generate_response(
    messages: list[AnyMessage],
    memory_context: UserInfo | None = None,
) -> AIMessage:
    """生成 Agent 的对话回复。

    将用户的消息历史连同 system prompt（包含记忆上下文）一起
    发送给主模型，生成个性化的回复。

    :param messages: 完整的对话消息历史列表。
    :param memory_context: 从长期记忆加载的用户信息，用于个性化回复。
    :return: 模型生成的 AI 回复消息。
    """
    # 构建带有记忆上下文的 system prompt
    system_message = _build_system_prompt(memory_context)

    # 将 system prompt 放在消息列表最前面，然后拼接对话历史
    full_messages = [system_message] + messages

    # 调用主模型生成回复
    primary_llm = get_primary_llm()
    return await primary_llm.ainvoke(full_messages)
