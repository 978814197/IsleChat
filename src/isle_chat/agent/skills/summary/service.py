"""
对话摘要服务模块。

提供对话历史的滑动窗口摘要功能：
1. 双条件判断：token 容量为主 + 消息条数兜底
2. 增量摘要：基于已有摘要做更新，不重复总结旧内容
3. 保留区裁剪：按 token 数从后往前保留消息，确保从 human 消息开始

使用分析模型（analyzer_model）生成摘要以降低成本。
"""

from langchain.messages import AnyMessage, HumanMessage
from langchain_core.messages import RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately

from ...core.settings import settings
from ...models.llm import get_analyzer_llm

# ── 摘要 prompt 模板 ──

_FIRST_SUMMARY_PROMPT = (
    "请总结以下对话的关键信息，包括讨论的话题、用户的情绪、重要事件等。\n"
    "要求：\n"
    "- 用简洁的中文概括\n"
    "- 保持信息密度，不要有废话\n"
    "- 优先保留对后续对话有用的信息"
)

_UPDATE_SUMMARY_PROMPT = (
    "以下是之前对话的摘要：\n{existing_summary}\n\n"
    "请结合以下新的对话内容，更新并扩展这份摘要。\n"
    "要求：\n"
    "- 用简洁的中文概括\n"
    "- 优先保留最近的、最重要的信息\n"
    "- 如果空间不够，可以舍弃最早的、最不重要的细节\n"
    "- 保持信息密度，不要有废话"
)


def should_summarize(messages: list[AnyMessage]) -> bool:
    """判断是否需要触发对话摘要。

    双条件判断：
    1. token 容量为主：消息历史 token 数超过消息预算的触发阈值
    2. 消息条数兜底：消息条数超过兜底上限

    :param messages: 当前对话消息列表。
    :return: 是否需要触发摘要。
    """
    ctx = settings.context

    # 条件一：token 容量判断
    messages_budget = int(ctx.max_context_tokens * ctx.messages_max_percentage)
    trigger_tokens = int(messages_budget * ctx.summary_trigger_threshold)
    current_tokens = count_tokens_approximately(messages)
    if current_tokens > trigger_tokens:
        return True

    # 条件二：消息条数兜底
    if len(messages) > ctx.fallback_max_messages:
        return True

    return False


def _find_keep_boundary(messages: list[AnyMessage], keep_tokens: int) -> int:
    """从后往前累计 token，找到保留区的分界索引。

    确保裁剪后的保留区从 HumanMessage 开始，避免从 AI 回复中间切断。

    :param messages: 消息列表。
    :param keep_tokens: 保留区的 token 预算。
    :return: 保留区起始索引（该索引及之后的消息被保留）。
    """
    total = 0
    boundary = 0  # 默认保留全部

    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = count_tokens_approximately([messages[i]])
        if total + msg_tokens > keep_tokens:
            # 超出预算，从 i+1 开始保留
            # 但要确保从 human 消息开始
            for j in range(i + 1, len(messages)):
                if isinstance(messages[j], HumanMessage):
                    return j
            return i + 1
        total += msg_tokens

    return boundary


async def summarize_and_trim(
    messages: list[AnyMessage],
    existing_summary: str,
    last_analyzed_index: int,
) -> dict:
    """执行对话摘要并裁剪旧消息。

    流程：
    1. 计算保留区边界（按 token 预算从后往前保留）
    2. 对保留区之前的旧消息生成增量摘要
    3. 通过 RemoveMessage 物理删除旧消息
    4. 同步调整分析指针 last_analyzed_index

    :param messages: 当前完整消息列表。
    :param existing_summary: 已有的对话摘要（可能为空字符串）。
    :param last_analyzed_index: 当前的分析指针位置。
    :return: 状态更新字典，包含 summary、messages（RemoveMessage 列表）、
             last_analyzed_index。
    """
    ctx = settings.context

    # 计算保留区 token 预算：摘要后保留的消息占消息预算的一半
    messages_budget = int(ctx.max_context_tokens * ctx.messages_max_percentage)
    keep_tokens = messages_budget // 2

    # 找到保留区边界
    boundary = _find_keep_boundary(messages, keep_tokens)

    # 如果没有需要摘要的消息（全部都在保留区内），直接返回
    if boundary == 0:
        return {}

    messages_to_summarize = messages[:boundary]

    # 构建摘要 prompt
    if existing_summary:
        prompt_text = _UPDATE_SUMMARY_PROMPT.format(
            existing_summary=existing_summary,
        )
    else:
        prompt_text = _FIRST_SUMMARY_PROMPT

    # 用分析模型生成摘要（便宜）
    analyzer = get_analyzer_llm()
    summary_messages = [
        *messages_to_summarize,
        HumanMessage(content=prompt_text),
    ]
    response = await analyzer.ainvoke(
        summary_messages,
        max_tokens=ctx.summary_max_tokens,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    # 通过 RemoveMessage 物理删除旧消息
    delete_msgs = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    # 同步调整分析指针：删除了 boundary 条消息，指针需要回退。
    # 当 last_analyzed_index < boundary 时（指针还没推进到被删除的范围），
    # 结果为 0，表示所有剩余消息都未被分析过，下次分析从头开始。
    new_analyzed_index = max(0, last_analyzed_index - boundary)

    return {
        "summary": response.content,
        "messages": delete_msgs,
        "last_analyzed_index": new_analyzed_index,
    }
