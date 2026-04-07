"""
记忆分析器模块。

负责分析对话内容，判断是否包含适合写入长期记忆的信息。
采用三层优化策略降低成本：

1. 规则预筛：零成本的关键词匹配，快速过滤明显不需要分析的对话
2. 合并分析：将用户信息和 Agent 配置的提取合并为一次 LLM 调用
3. 兜底分析：每 N 轮强制分析一次，防止规则预筛遗漏信息

使用较小的分析模型（analyzer_llm）以降低成本和延迟。
"""

import json
import re

from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from ...models.llm import get_analyzer_llm
from ...models.schemas import ConversationAnalysisResult

# ── 兜底分析间隔（每 N 轮强制分析一次）──
FALLBACK_INTERVAL = 5

# ── 规则预筛：用户信息相关的触发模式 ──
_USER_INFO_PATTERNS = re.compile(
    r"我叫|我是|我在|我住|我的名字|名字是|我做|我的工作|工作是|"
    r"我喜欢|我的爱好|我的职业|职业是|岁了|在.*工作|在.*上班|"
    r"住在|来自|老家|家乡|毕业|专业是|学的是"
)

# ── 规则预筛：Agent 配置相关的触发模式 ──
_AGENT_PROFILE_PATTERNS = re.compile(
    r"你叫|叫你|你的名字|以后叫|改个名|你的性格|说话风格|"
    r"称呼我|叫我|喊我|你要|你得|你应该|你是.*的|"
    r"设定|人设|风格|语气|撒娇|可爱|温柔"
)

# ── 统一分析器的 system prompt 模板 ──
_UNIFIED_ANALYZER_PROMPT = (
    "你是对话信息提取器，需要同时完成两项分析任务：\n\n"
    "【任务一：用户信息提取】\n"
    "判断对话中是否包含适合写入长期记忆的稳定用户信息，"
    "例如姓名、城市、职业、长期偏好。"
    "不要提取一次性的临时请求（如「帮我查个天气」中的城市不算长期信息）。\n\n"
    "【任务二：Agent 配置提取】\n"
    "判断对话中用户是否在设置或修改 Agent 的身份信息，"
    "例如给 Agent 起名字、设定 Agent 的性格、指定 Agent 如何称呼用户、"
    "设定 Agent 的说话风格、或给出其他自定义指令。"
    "只提取用户明确表达的设置意图，不要猜测或推断。\n\n"
    "返回的 JSON 结构如下:\n{schema}"
)


def should_trigger_analysis(
    user_message: str,
    turns_since_last_analysis: int,
) -> bool:
    """判断是否需要触发 LLM 分析。

    三层判断逻辑：
    1. 规则预筛命中 → 触发分析
    2. 兜底计数器达到阈值 → 触发分析
    3. 都未命中 → 跳过分析

    :param user_message: 用户最新一条消息的文本内容。
    :param turns_since_last_analysis: 距上次分析经过的轮次数。
    :return: 是否需要触发分析。
    """
    # 规则预筛：关键词匹配
    if _USER_INFO_PATTERNS.search(user_message):
        return True
    if _AGENT_PROFILE_PATTERNS.search(user_message):
        return True

    # 兜底分析：每 N 轮强制触发
    if turns_since_last_analysis >= FALLBACK_INTERVAL:
        return True

    return False


async def analyze_conversation(
    messages: list[AnyMessage],
) -> ConversationAnalysisResult:
    """统一分析对话，同时判断用户信息和 Agent 配置的提取需求。

    将用户信息提取和 Agent 配置提取合并为一次 LLM 调用，
    减少 API 调用次数和延迟。

    :param messages: 待分析的消息列表（从上次分析位置到当前的所有消息）。
    :return: 统一分析结果，包含用户信息和 Agent 配置的提取判断。
    """
    # 构造 system prompt，将 ConversationAnalysisResult 的 JSON Schema 嵌入其中
    schema_json = json.dumps(
        ConversationAnalysisResult.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    system_message = SystemMessage(
        content=_UNIFIED_ANALYZER_PROMPT.format(schema=schema_json),
    )

    # 调用分析模型，要求返回 JSON 格式
    analyzer_llm = get_analyzer_llm()
    response = await analyzer_llm.ainvoke(
        [system_message] + messages,
        response_format={"type": "json_object"},
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    # 将 LLM 的 JSON 输出解析为结构化对象
    return ConversationAnalysisResult.model_validate_json(response.content)


def get_last_user_message(messages: list[AnyMessage]) -> str:
    """从消息列表中获取最后一条用户消息的文本内容。

    :param messages: 消息列表。
    :return: 最后一条用户消息的文本，若不存在则返回空字符串。
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
