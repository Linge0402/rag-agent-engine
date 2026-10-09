"""KnowledgePack 协议——领域包接入引擎的唯一入口。

一个领域（法律法规、技术文档……）要接入本引擎，唯一要做的事情是
实现本协议：声明自己的解析器、切分策略、可过滤字段、Agent 工具、
提示词模板与评估集。引擎侧（core/）不感知任何领域细节。

这是"引擎 + 领域包"架构的核心抽象：packs/devdocs 的验收标准
"不改 core 一行代码接入新领域"，约束的就是本协议的完备性。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.agent.tools import Tool
from core.chunking.base import ChunkStrategy
from core.parsing.base import Parser
from core.service.schemas import EvalSet, FilterSchema, PromptTemplate


@runtime_checkable
class KnowledgePack(Protocol):
    """领域包协议。六个声明，缺一不可：

    - parsers()        领域文档如何变成 ParsedDocument
    - chunk_strategy() 文档如何切分（领域结构，如法规"编-章-节-条"）
    - filter_schema()  哪些 metadata 字段允许硬过滤（如"现行有效"）——
                       声明之外的字段，检索层一律拒绝作为过滤条件
    - tools()          暴露给 Agent 的领域工具
    - prompt_template() 领域 grounded 模板，领域硬约束写在这里
    - eval_sets()      领域评估集；没有评估集的 pack 不允许合入
    """

    name: str  # 全局唯一包名，如 "legal" / "devdocs"

    def parsers(self) -> list[Parser]: ...

    def chunk_strategy(self) -> ChunkStrategy: ...

    def filter_schema(self) -> FilterSchema: ...

    def tools(self) -> list[Tool]: ...

    def prompt_template(self) -> PromptTemplate: ...

    def eval_sets(self) -> list[EvalSet]: ...


def validate_pack(pack: KnowledgePack) -> list[str]:
    """对一个 pack 做静态自检，返回问题列表（空列表表示通过）。

    供 `scripts/` 与 CI 使用：接入新领域时先过这一关再谈跑 demo。
    """
    problems: list[str] = []

    if not pack.name or not pack.name.isidentifier():
        problems.append(f"pack 名非法: {pack.name!r}")

    if not pack.parsers():
        problems.append("parsers() 不能为空：没有解析器就无法入库")

    strategy = pack.chunk_strategy()
    if not strategy.name:
        problems.append("chunk_strategy() 返回的策略缺少 name")

    schema = pack.filter_schema()
    names = [f.name for f in schema.fields]
    if len(names) != len(set(names)):
        problems.append(f"filter_schema() 存在重复字段声明: {names}")

    tool_names = [t.name for t in pack.tools()]
    if len(tool_names) != len(set(tool_names)):
        problems.append(f"tools() 存在重名工具: {tool_names}")

    template = pack.prompt_template()
    for placeholder in ("{context}", "{question}"):
        if placeholder not in template.user:
            problems.append(f"prompt_template().user 缺少占位符 {placeholder}")

    eval_sets = pack.eval_sets()
    if not eval_sets or all(not s.cases for s in eval_sets):
        problems.append("eval_sets() 为空：没有评估集的 pack 不允许合入")

    return problems
