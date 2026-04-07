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
from ...models.schemas import AgentProfileExtractionResult, UserInfoExtractionResult

# ── 用户信息分析器的 system prompt 模板 ──
_USER_INFO_ANALYZER_PROMPT = (
    "你是用户信息提取器。"
    "请判断最近一轮对话中是否包含适合写入长期记忆的稳定用户信息，"
    "例如姓名、城市、职业、长期偏好。"
    "不要提取一次性的临时请求（如「帮我查个天气」中的城市不算长期信息）。"
    "返回的 JSON 结构如下:\n{schema}"
)

# ── Agent 配置分析器的 system prompt 模板 ──
_AGENT_PROFILE_ANALYZER_PROMPT = (
    "你是 Agent 配置提取器。"
    "请判断最近一轮对话中，用户是否在设置或修改 Agent 的身份信息，"
    "例如给 Agent 起名字、设定 Agent 的性格、指定 Agent 如何称呼用户、"
    "设定 Agent 的说话风格、或给出其他自定义指令。"
    "只提取用户明确表达的设置意图，不要猜测或推断。"
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
        content=_USER_INFO_ANALYZER_PROMPT.format(schema=schema_json),
    )

    # 调用分析模型，要求返回 JSON 格式
    analyzer_llm = get_analyzer_llm()
    response = await analyzer_llm.ainvoke(
        [system_message] + latest_turn,
        response_format={"type": "json_object"},
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    # 将 LLM 的 JSON 输出解析为结构化对象
    return UserInfoExtractionResult.model_validate_json(response.content)


async def analyze_agent_profile(latest_turn: list[AnyMessage]) -> AgentProfileExtractionResult:
    """分析最近一轮对话，判断是否包含用户对 Agent 身份/行为的设置指令。

    将对话内容发送给分析模型，由 LLM 判断用户是否在设置 Agent 的
    名字、性格、称呼方式、说话风格或其他自定义指令。

    :param latest_turn: 最近一轮对话消息列表（通常为一问一答两条消息）。
    :return: 提取结果，包含是否需要提取的标志和提取出的 Agent 配置。
    """
    # 构造 system prompt，将 AgentProfileExtractionResult 的 JSON Schema 嵌入其中
    schema_json = json.dumps(
        AgentProfileExtractionResult.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    system_message = SystemMessage(
        content=_AGENT_PROFILE_ANALYZER_PROMPT.format(schema=schema_json),
    )

    # 调用分析模型，要求返回 JSON 格式
    analyzer_llm = get_analyzer_llm()
    response = await analyzer_llm.ainvoke(
        [system_message] + latest_turn,
        response_format={"type": "json_object"},
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    # 将 LLM 的 JSON 输出解析为结构化对象
    return AgentProfileExtractionResult.model_validate_json(response.content)
