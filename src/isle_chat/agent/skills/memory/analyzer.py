"""
记忆分析器模块。

负责分析对话内容，判断是否包含适合写入长期记忆的稳定用户信息。
使用较小的分析模型（analyzer_llm）以降低成本和延迟。

分析逻辑：
1. 构造一个专用的 system prompt，指导 LLM 识别稳定用户信息
2. 将最近一轮对话发送给分析模型
3. 解析结构化的 JSON 输出，返回提取结果
"""

import json

from langchain.messages import AnyMessage, SystemMessage

from ...models.llm import get_analyzer_llm
from ...models.schemas import UserInfoExtractionResult


# 分析器的 system prompt 模板
_ANALYZER_SYSTEM_PROMPT = (
    "你是用户信息提取器。"
    "请判断最近一轮对话中是否包含适合写入长期记忆的稳定用户信息，"
    "例如姓名、城市、职业、长期偏好。"
    "不要提取一次性的临时请求（如「帮我查个天气」中的城市不算长期信息）。"
    "返回的 JSON 结构如下:\n{schema}"
)


async def analyze_conversation(latest_turn: list[AnyMessage]) -> UserInfoExtractionResult:
    """分析最近一轮对话，判断是否包含需要提取的用户信息。

    将对话内容发送给分析模型，由 LLM 判断对话中是否包含
    姓名、城市、职业、偏好等稳定的用户信息。

    :param latest_turn: 最近一轮对话消息列表（通常为一问一答两条消息）。
    :return: 提取结果，包含是否需要提取的标志和提取出的用户信息。
    """
    # 构造 system prompt，将 UserInfoExtractionResult 的 JSON Schema 嵌入其中，
    # 让 LLM 知道需要返回的数据结构
    schema_json = json.dumps(
        UserInfoExtractionResult.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    system_message = SystemMessage(
        content=_ANALYZER_SYSTEM_PROMPT.format(schema=schema_json),
    )

    # 调用分析模型，要求返回 JSON 格式
    analyzer_llm = get_analyzer_llm()
    response = await analyzer_llm.ainvoke(
        [system_message] + latest_turn,
        response_format={"type": "json_object"},
    )

    # 将 LLM 的 JSON 输出解析为结构化对象
    return UserInfoExtractionResult.model_validate_json(response.content)
