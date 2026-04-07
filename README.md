# 🏝️ IsleChat

**基于 LangGraph 的长期陪伴型聊天 Agent**

IsleChat 是一个温暖、友善的 AI 聊天伙伴，能够记住你的姓名、偏好和习惯，在每一次对话中提供个性化的陪伴体验。

---

## ✨ 特性

- **长期记忆** — 自动识别并记住用户的姓名、城市、职业、兴趣偏好等稳定信息，跨会话持久化
- **短期记忆** — 基于 LangGraph Checkpointer 的对话历史管理，同一线程内自动延续上下文
- **个性化回复** — 将用户画像注入 System Prompt，让每次对话都带有专属感
- **双模型架构** — 主模型负责对话回复，分析模型负责信息提取，兼顾质量与成本
- **对话历史摘要** — 滑动窗口 + 增量摘要，双条件触发（token 容量 + 消息条数），确保长期对话不超出上下文窗口
- **三层分析优化** — 规则预筛 → 合并分析 → 兜底分析，降低约 80-90% 分析模型调用成本
- **可扩展技能系统** — 模块化的 Skills 层设计，方便后续接入天气查询、日程管理等新能力
- **PostgreSQL 持久化** — 使用 LangGraph 官方的 `langgraph-checkpoint-postgres` 组件，数据安全可靠

## 🏗️ 架构概览

IsleChat 基于 [LangGraph](https://langchain-ai.github.io/langgraph/) StateGraph 构建，采用分层架构：

```
┌──────────────────────────────────────────────────────┐
│                     入口层                            │
│                  __main__.py                          │
├──────────────────────────────────────────────────────┤
│                   状态图层                            │
│          graph/ (builder + nodes + edges)             │
├──────────────────────────────────────────────────────┤
│                   技能层                              │
│       skills/memory/    skills/response/              │
├──────────────────────────────────────────────────────┤
│                  基础设施层                            │
│        models/ (LLM)    core/ (settings, db)         │
└──────────────────────────────────────────────────────┘
```

### Agent 执行流程

```
START
  │
  ▼
load_memory                        ← 从 Store 加载用户画像 + Agent 配置
  │
  ▼
should_summarize?                  ← 双条件判断（token 容量 + 消息条数）
  ├─(需要)→ summarize_conversation ← 增量摘要 + 物理删除旧消息
  │              │
  │              ▼
  └─(不需要)→ generate_assistant_response ← 生成 AI 回复（带摘要 + 记忆上下文）
                 │
                 ▼
             check_and_analyze     ← 三层记忆分析优化：
                 │                    1. 规则预筛（零成本关键词匹配）
                 │                    2. 合并分析（一次 LLM 调用）
                 │                    3. 兜底分析（每 N 轮强制触发）
                 │
                 ├─(需要保存用户信息)──→ save_user_info ──→ END
                 ├─(需要保存Agent配置)─→ save_agent_profile ──→ END
                 └─(都不需要)─────────→ END
```

## 📦 安装

### 前置要求

- Python >= 3.12
- PostgreSQL（用于短期/长期记忆持久化）
- [uv](https://docs.astral.sh/uv/)（推荐的 Python 包管理工具）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/ynyg/IsleChat.git
cd IsleChat

# 使用 uv 安装依赖
uv sync
```

## ⚙️ 配置

IsleChat 通过环境变量进行配置，使用 `__`（双下划线）分隔嵌套层级。

### 必要的环境变量

```bash
# ── 数据库配置 ──
DB__HOST=localhost          # PostgreSQL 主机地址
DB__PORT=5432               # PostgreSQL 端口号
DB__USERNAME=postgres       # 数据库用户名
DB__PASSWORD=your_password  # 数据库密码
DB__NAME=isle_chat          # 数据库名称

# ── LLM 配置 ──
LLM__BASE_URL=https://api.openai.com/v1   # LLM API 地址（支持 OpenAI 兼容接口）
LLM__API_KEY=sk-xxx                        # LLM API 密钥
LLM__PRIMARY_MODEL=gpt-4o-mini             # 主模型（用于对话回复）
LLM__ANALYZER_MODEL=gpt-4o-mini            # 分析模型（用于信息提取，可用更小的模型降低成本）
```

你可以将这些变量写入 `.env` 文件或直接导出到系统环境中。

### 数据库准备

确保 PostgreSQL 已启动，并创建对应的数据库：

```sql
CREATE DATABASE isle_chat;
```

> 应用首次启动时会自动创建所需的表结构（checkpointer 和 store 表），无需手动建表。

## 🚀 使用方法

```bash
# 运行 Agent
uv run python -m isle_chat
```

或者通过安装后的命令行入口：

```bash
uv run isle-chat
```

## 📁 项目结构

```
IsleChat/
├── pyproject.toml                          # 项目配置与依赖管理
├── README.md                               # 项目说明文档
├── uv.lock                                 # 依赖锁定文件
└── src/
    └── isle_chat/
        ├── __init__.py                     # 包入口
        ├── __main__.py                     # 应用入口，生命周期管理
        │
        ├── core/                           # 基础设施层
        │   ├── settings.py                 # 应用配置（pydantic-settings）
        │   └── database.py                 # 数据库连接管理（checkpointer + store）
        │
        └── agent/                          # Agent 核心模块
            ├── state/                      # 状态定义
            │   ├── state.py                # AgentState — 图中流转的核心状态
            │   └── context.py              # AgentContext — 运行时只读上下文
            │
            ├── graph/                      # LangGraph 状态图
            │   ├── builder.py              # 图构建与编译
            │   ├── edges.py                # 条件边路由函数
            │   └── nodes/                  # 图节点（薄节点，委托 skills 处理）
            │       ├── response.py         # 回复生成节点
            │       └── memory.py           # 记忆分析与保存节点
            │
            ├── models/                     # 模型层
            │   ├── llm.py                  # LLM 实例工厂（懒加载）
            │   └── schemas.py              # 共享的 Pydantic 数据模型
            │
            └── skills/                     # 可扩展技能层
                ├── memory/                 # 记忆技能
                │   ├── types.py            # 记忆相关类型定义
                │   ├── analyzer.py         # 对话分析器（规则预筛 + 合并分析 + 兜底机制）
                │   ├── extractor.py        # 信息提取器（预留扩展）
                │   ├── repository.py       # 记忆持久化仓库（LangGraph Store）
                │   └── service.py          # 记忆服务门面
                ├── summary/                # 对话摘要技能
                │   └── service.py          # 滑动窗口摘要服务（双条件触发 + 增量摘要）
                └── response/               # 回复生成技能
                    └── service.py          # 回复生成服务（含 System Prompt 构建）
```

## 🔌 扩展新技能

IsleChat 的技能系统设计为可插拔的模块化架构，添加新技能只需以下步骤：

1. **创建技能子包** — 在 `skills/` 下新建目录，例如 `skills/weather/`
2. **定义类型** — 在 `types.py` 中定义技能相关的数据类型
3. **实现服务** — 在 `service.py` 中实现核心逻辑，对外提供统一接口
4. **创建图节点** — 在 `graph/nodes/` 下新建节点文件，调用技能服务
5. **注册到图中** — 在 `graph/builder.py` 中将新节点和边注册到状态图

```
skills/weather/
├── __init__.py         # 导出服务接口
├── types.py            # WeatherInfo 等类型定义
└── service.py          # 天气查询服务
```

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM 集成 | [LangChain](https://python.langchain.com/) + OpenAI 兼容接口 |
| 短期记忆 | LangGraph AsyncPostgresSaver（Checkpointer） + 对话摘要 |
| 长期记忆 | LangGraph AsyncPostgresStore（Key-Value Store） |
| 数据库驱动 | [psycopg](https://www.psycopg.org/)（异步模式） |
| 配置管理 | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| 包管理 | [uv](https://docs.astral.sh/uv/) |

## 📄 License

[MIT](LICENSE)
