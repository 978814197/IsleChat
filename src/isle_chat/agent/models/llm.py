"""
LLM 模型实例工厂模块。

采用懒加载模式创建 LLM 实例，避免在模块导入时就触发配置读取，
便于测试和按需初始化。

使用方式::

    from isle_chat.agent.models.llm import get_primary_llm, get_analyzer_llm

    llm = get_primary_llm()
    response = await llm.ainvoke(messages)
"""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from ...core.settings import settings


def _build_chat_model(model_name: str) -> BaseChatModel:
    """根据模型名称构建聊天模型实例。

    使用 langchain 的 init_chat_model 工厂方法创建模型，
    所有模型共享同一组 base_url 和 api_key 配置。

    :param model_name: 模型名称，例如 "gpt-4o-mini"。
    :return: 初始化完成的聊天模型实例。
    """
    llm_settings = settings.llm
    return init_chat_model(
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key.get_secret_value(),
        model=model_name,
        model_provider="openai",
        reasoning_effort="low",
    )


@lru_cache
def get_primary_llm() -> BaseChatModel:
    """获取主聊天模型实例（懒加载 + 缓存）。

    主模型用于生成对话回复，通常使用较大、能力较强的模型。

    :return: 主聊天模型实例。
    """
    return _build_chat_model(settings.llm.primary_model)


@lru_cache
def get_analyzer_llm() -> BaseChatModel:
    """获取分析模型实例（懒加载 + 缓存）。

    分析模型用于用户信息提取、意图分析等轻量任务，
    可以使用更小更便宜的模型以降低成本。

    :return: 分析模型实例。
    """
    return _build_chat_model(settings.llm.analyzer_model)
