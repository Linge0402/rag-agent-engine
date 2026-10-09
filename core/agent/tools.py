"""Agent 工具协议与调用上下文。

引擎内置通用工具（knowledge_search / get_chunk）；
领域工具（时效校验、版本查询、沙箱执行等）由领域包实现本协议，
经 KnowledgePack.tools() 注册进 Agent 的工具表。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from core.service.schemas import Chunk, FilterSchema


class QueryContext(BaseModel):
    """一次提问的上下文，随工具调用透传。

    pack_filter_schema 让通用工具知道当前知识库有哪些字段可过滤；
    session 存放多轮对话状态（由 core/agent/memory.py 管理）。
    """

    session_id: str
    pack_name: str
    pack_filter_schema: FilterSchema = Field(default_factory=FilterSchema)
    session: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """工具调用的结构化结果。

    chunks 用于检索类工具返回命中文段；data 承载领域工具的
    任意结构化输出（如时效校验结论）；error 非空表示调用失败，
    Agent 应把失败原因纳入后续决策而非直接抛异常。
    """

    chunks: list[Chunk] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Tool(Protocol):
    """Agent 可调用工具。

    实现要求：
    - description 写给 LLM 看，须说清"何时该用我、参数怎么填"；
    - args_schema 是参数的唯一校验入口，run() 收到的必是校验后的实例；
    - run() 不抛未捕获异常，失败统一走 ToolResult.error。
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult: ...
