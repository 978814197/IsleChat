"""
记忆技能类型定义模块。

定义记忆系统中使用的内部数据类型。
与 models/schemas.py 中的 UserInfo 不同，这里定义的是
记忆系统内部的分析和传输类型。

注意：
    用户记忆的持久化存储使用 LangGraph 官方的 AsyncPostgresStore，
    数据以 namespace + key-value 形式组织，无需自定义表结构。
    详见 skills/memory/repository.py 中的存储约定。
"""
