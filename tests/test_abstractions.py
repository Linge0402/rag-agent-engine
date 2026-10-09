"""核心抽象测试：用最小 Fake 实现验证三个协议的可用性与自检逻辑。

Fake 组件是本项目测试基建的一部分：embedding / 向量库 / LLM 在
单元测试中一律用 Fake 注入，保证测试离线可跑。
"""

from pathlib import Path

from pydantic import BaseModel

from core.agent.tools import QueryContext, Tool, ToolResult
from core.chunking.base import ChunkStrategy
from core.pack import KnowledgePack, validate_pack
from core.parsing.base import Parser
from core.service.schemas import (
    Chunk,
    ChunkMetadata,
    EvalCase,
    EvalSet,
    FilterField,
    FilterSchema,
    ParsedDocument,
    PromptTemplate,
)

# ---------------------------------------------------------------------------
# 最小 Fake 实现
# ---------------------------------------------------------------------------


class FakeParser:
    name = "fake-md"

    def supports(self, path: Path) -> bool:
        return path.suffix == ".md"

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(doc_id=path.stem, source=str(path), text=path.read_text())


class FakeChunkStrategy(ChunkStrategy):
    @property
    def name(self) -> str:
        return "fake-whole-doc"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}-c1",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=ChunkMetadata(source=doc.source),
            )
        ]


class SearchArgs(BaseModel):
    query: str
    top_k: int = 5


class FakeSearchTool:
    name = "knowledge_search"
    description = "在知识库中检索与 query 相关的文段。"
    args_schema = SearchArgs

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, SearchArgs)
        return ToolResult(data={"query": args.query, "hits": 0})


class FakePack:
    name = "fakepack"

    def __init__(self, eval_cases: list[EvalCase] | None = None):
        self._eval_cases = eval_cases if eval_cases is not None else [EvalCase(question="q")]

    def parsers(self):
        return [FakeParser()]

    def chunk_strategy(self):
        return FakeChunkStrategy()

    def filter_schema(self):
        return FilterSchema(
            fields=[FilterField(name="validity", type="keyword", description="时效状态")]
        )

    def tools(self):
        return [FakeSearchTool()]

    def prompt_template(self):
        return PromptTemplate(system="sys", user="{context}\n{question}")

    def eval_sets(self):
        return [EvalSet(name="fake", cases=self._eval_cases)]


# ---------------------------------------------------------------------------
# 协议符合性
# ---------------------------------------------------------------------------


def test_protocol_conformance():
    assert isinstance(FakeParser(), Parser)
    assert isinstance(FakeSearchTool(), Tool)
    assert isinstance(FakePack(), KnowledgePack)


def test_incomplete_pack_fails_protocol_check():
    class Incomplete:
        name = "incomplete"

    assert not isinstance(Incomplete(), KnowledgePack)


def test_chunk_strategy_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        ChunkStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 抽象的行为契约
# ---------------------------------------------------------------------------


def test_strategy_produces_traceable_chunks(tmp_path: Path):
    doc_file = tmp_path / "民法典节选.md"
    doc_file.write_text("第一百八十八条 向人民法院请求保护民事权利的诉讼时效期间为三年。")

    parser = FakeParser()
    assert parser.supports(doc_file)

    doc = parser.parse(doc_file)
    chunks = FakeChunkStrategy().chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_id.startswith(doc.doc_id)
    assert chunks[0].metadata.source == str(doc_file)


async def test_tool_run_contract():
    tool = FakeSearchTool()
    ctx = QueryContext(session_id="s1", pack_name="fakepack")
    result = await tool.run(SearchArgs(query="诉讼时效"), ctx)
    assert result.ok
    assert result.error is None


def test_tool_result_failure_channel():
    result = ToolResult(error="qdrant 连接超时")
    assert not result.ok


# ---------------------------------------------------------------------------
# validate_pack 自检
# ---------------------------------------------------------------------------


def test_validate_pack_pass():
    assert validate_pack(FakePack()) == []


def test_validate_pack_rejects_empty_eval_set():
    problems = validate_pack(FakePack(eval_cases=[]))
    assert any("eval_sets" in p for p in problems)


def test_validate_pack_rejects_missing_placeholder():
    class BadTemplatePack(FakePack):
        def prompt_template(self):
            return PromptTemplate(system="sys", user="只有上下文 {context}")

    problems = validate_pack(BadTemplatePack())
    assert any("{question}" in p for p in problems)


def test_validate_pack_rejects_duplicate_filter_fields():
    class DupFilterPack(FakePack):
        def filter_schema(self):
            return FilterSchema(
                fields=[
                    FilterField(name="validity", type="keyword"),
                    FilterField(name="validity", type="keyword"),
                ]
            )

    problems = validate_pack(DupFilterPack())
    assert any("重复字段" in p for p in problems)
