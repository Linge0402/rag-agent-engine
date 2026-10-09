"""核心数据契约。

本模块定义贯穿入库与查询两条流水线的全部数据结构。
领域包（packs/）只允许通过 `ChunkMetadata.extra` 与各自的 FilterSchema 扩展，
不允许修改本文件中的基础契约。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 入库侧契约
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Chunk 的可过滤元数据。

    前 6 个字段为引擎内置的通用维度；领域特有字段（如法规的效力级别、
    技术文档的版本号）放入 extra，并在所属 KnowledgePack 的
    filter_schema 中声明后才允许用于硬过滤。
    """

    source: str  # 来源标识：文件路径 / URL / 法规名称
    # 结构层级，如 ["第一编", "第一章", "第三条"]
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    version: str | None = None  # 文档 / 法规版本
    validity: str | None = None  # 时效状态，如 "现行有效" / "已修订" / "已废止"
    acl: list[str] = Field(default_factory=list)  # 可见范围；空列表表示公开
    extra: dict[str, Any] = Field(default_factory=dict)  # 领域扩展字段


class Chunk(BaseModel):
    """切分后的最小检索单元。chunk_id 全局唯一且可追溯。"""

    chunk_id: str  # 可追溯 ID，如 "doc-001-p3-c2"
    doc_id: str
    text: str
    metadata: ChunkMetadata


class ParsedDocument(BaseModel):
    """解析阶段的输出，切分阶段的输入。"""

    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 查询侧契约
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """一条结论的出处：挂到具体 chunk，并附原文片段供核验。"""

    chunk_id: str
    quote: str  # 原文片段（须为被引用 chunk 的逐字子串）
    source: str | None = None


class Answer(BaseModel):
    """引擎对一次提问的最终响应。"""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["answered", "refused", "needs_review"]
    latency_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 过滤 DSL 声明（检索硬过滤的 schema 部分；求值器在 core/retrieval/filters.py）
# ---------------------------------------------------------------------------


class FilterField(BaseModel):
    """一个可用于硬过滤的 metadata 字段声明。"""

    name: str  # ChunkMetadata 字段名，或 "extra.<key>" 形式
    type: Literal["keyword", "integer", "float", "bool", "datetime"]
    description: str = ""


class FilterSchema(BaseModel):
    """领域包声明的全部可过滤字段。"""

    fields: list[FilterField] = Field(default_factory=list)

    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}


# ---------------------------------------------------------------------------
# 提示词与评估集契约
# ---------------------------------------------------------------------------


class PromptTemplate(BaseModel):
    """grounded 提示词模板。由领域包提供，引擎渲染。

    user 模板必须包含 {context} 与 {question} 两个占位符；
    领域的硬约束（如"引用条文必须现行有效"）写在 system 中。
    """

    system: str
    user: str

    def render_user(self, context: str, question: str) -> str:
        return self.user.format(context=context, question=question)


class EvalCase(BaseModel):
    """单条评估样本。三列断言可按需组合。"""

    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)  # 检索断言：应命中的 chunk
    expected_answer_contains: list[str] = Field(default_factory=list)  # 生成断言：答案应包含的要点
    should_refuse: bool = False  # 拒答断言：边界/越界问题应被拒答
    notes: str = ""


class EvalSet(BaseModel):
    """一个领域的评估集。"""

    name: str
    cases: list[EvalCase] = Field(default_factory=list)
