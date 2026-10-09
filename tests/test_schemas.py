"""数据契约测试：公开领域语料（法规条文）作为示例数据。"""

import pytest
from pydantic import ValidationError

from core.service.schemas import (
    Answer,
    Chunk,
    ChunkMetadata,
    Citation,
    EvalCase,
    EvalSet,
    FilterSchema,
    PromptTemplate,
)


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="minfadian-p1-c3",
        doc_id="minfadian",
        text="向人民法院请求保护民事权利的诉讼时效期间为三年。",
        metadata=ChunkMetadata(
            source="中华人民共和国民法典",
            section_path=["第一编 总则", "第九章 诉讼时效", "第一百八十八条"],
            validity="现行有效",
            extra={"效力级别": "法律", "发文机关": "全国人民代表大会"},
        ),
    )


def test_chunk_roundtrip():
    chunk = make_chunk()
    restored = Chunk.model_validate_json(chunk.model_dump_json())
    assert restored == chunk
    assert restored.metadata.section_path[-1] == "第一百八十八条"
    assert restored.metadata.extra["效力级别"] == "法律"


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(source="中华人民共和国民法典")
    assert meta.section_path == []
    assert meta.acl == []  # 空列表 = 公开
    assert meta.validity is None
    assert meta.extra == {}


def test_answer_confidence_bounds():
    chunk = make_chunk()
    citation = Citation(chunk_id=chunk.chunk_id, quote=chunk.text, source=chunk.metadata.source)
    answer = Answer(
        text="诉讼时效期间为三年。",
        citations=[citation],
        confidence=0.92,
        status="answered",
        latency_ms=340,
    )
    assert answer.status == "answered"

    with pytest.raises(ValidationError):
        Answer(text="x", confidence=1.5, status="answered", latency_ms=1)
    with pytest.raises(ValidationError):
        Answer(text="x", confidence=0.5, status="answered", latency_ms=-1)
    with pytest.raises(ValidationError):
        Answer(text="x", confidence=0.5, status="hallucinated", latency_ms=1)  # type: ignore[arg-type]


def test_refused_answer_needs_no_citations():
    answer = Answer(text="未检索到现行有效的相关规定。", confidence=0.3, status="refused", latency_ms=120)
    assert answer.citations == []


def test_prompt_template_render():
    template = PromptTemplate(
        system="你只能依据给定条文回答，且引用条文必须现行有效。",
        user="条文：\n{context}\n\n问题：{question}",
    )
    rendered = template.render_user(context="第一百八十八条……", question="诉讼时效是多久？")
    assert "第一百八十八条" in rendered and "诉讼时效是多久？" in rendered


def test_filter_schema_unique_names():
    schema = FilterSchema(fields=[])
    assert schema.field_names() == set()


def test_eval_set_with_refusal_case():
    eval_set = EvalSet(
        name="legal-validity",
        cases=[
            EvalCase(question="民法典诉讼时效多久？", expected_chunk_ids=["minfadian-p1-c3"]),
            EvalCase(question="已被废止的合同法怎么说？", should_refuse=True, notes="时效陷阱题"),
        ],
    )
    assert len(eval_set.cases) == 2
    assert eval_set.cases[1].should_refuse
