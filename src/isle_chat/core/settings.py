"""
应用配置模块。

使用 pydantic-settings 管理所有配置项，支持通过环境变量注入。
环境变量命名规则：使用双下划线 "__" 分隔嵌套层级，例如：
    LLM__BASE_URL=https://api.example.com
    LLM__API_KEY=sk-xxx
    DB__HOST=localhost
"""

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    """数据库配置。

    用于配置长期记忆等持久化存储所需的数据库连接信息。
    """

    host: str = Field(..., description="数据库的主机地址")
    port: int = Field(..., description="数据库的端口号")
    username: str = Field(..., description="数据库的用户名")
    password: SecretStr = Field(..., description="数据库的密码")
    name: str = Field(..., description="数据库的名称")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """构建 PostgreSQL 连接 DSN 字符串。

        格式: postgresql://username:password@host:port/name
        用于 asyncpg 连接池创建。
        """
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.username}:{pwd}@{self.host}:{self.port}/{self.name}"


class LLMSettings(BaseSettings):
    """LLM 模型配置。

    支持配置多个模型：主模型用于生成对话回复，分析模型用于
    用户信息提取等轻量分析任务（可使用更小更便宜的模型）。
    """

    base_url: str = Field(..., description="LLM 服务的 URL")
    api_key: SecretStr = Field(..., description="LLM 服务的 API Key")
    primary_model: str = Field(..., description="主模型名称，用于生成对话回复")
    analyzer_model: str = Field(..., description="分析模型名称，用于信息提取等轻量任务")


class ApplicationSettings(BaseSettings):
    """应用全局配置。

    通过 env_nested_delimiter="__" 支持嵌套环境变量，例如：
        LLM__BASE_URL, LLM__API_KEY, DB__HOST 等。
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    db: DBSettings = Field(default_factory=DBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


# 全局配置单例，在应用启动时自动从环境变量加载
settings = ApplicationSettings()
