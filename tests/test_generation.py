"""prompt 组装与 LLM 引用校验测试。"""

import httpx

from core.generation.llm import (
    LLMClient,
    OpenAIChatClient,
    build_answer,
    extract_cited_ids,
    validate_citations,
)
from core.generation.prompt import DEFAULT_TEMPLATE, build_context, build_messages
from core.service.schemas import Chunk, ChunkMetadata

CHUNKS = [
    Chunk(
        chunk_id="minfadian-c003",
        doc_id="minfadian",
        text="向人民法院请求保护民事权利的诉讼时效期间为三年。",
        metadata=ChunkMetadata(
            source="民法典.md",
            section_path=["第一编 总则", "第九章 诉讼时效"],
        ),
    ),
    Chunk(
        chunk_id="minfadian-c007",
        doc_id="minfadian",
        text="诉讼时效期间自权利人知道或者应当知道权利受到损害之日起计算。",
        metadata=ChunkMetadata(source="民法典.md", section_path=["第一编 总则"]),
    ),
]


# ---- prompt 组装 ----


def test_build_context_marks_chunk_ids_and_source():
    context = build_context(CHUNKS)
    assert "【minfadian-c003】" in context
    assert "来源：民法典.md" in context
    assert "第一编 总则 > 第九章 诉讼时效" in context


def test_build_messages_uses_template():
    messages = build_messages(DEFAULT_TEMPLATE, CHUNKS, "诉讼时效是多久？")
    assert messages[0]["role"] == "system"
    assert "不得编造" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "【minfadian-c003】" in messages[1]["content"]
    assert "诉讼时效是多久？" in messages[1]["content"]


# ---- 引用提取与 Answer 组装 ----


def test_extract_cited_ids_ordered_dedup():
    text = "期间为三年【minfadian-c003】，起算点另算【minfadian-c007】，重申【minfadian-c003】。"
    assert extract_cited_ids(text) == ["minfadian-c003", "minfadian-c007"]


def test_build_answer_with_valid_citations():
    answer = build_answer("期间为三年【minfadian-c003】。", CHUNKS, latency_ms=100)
    assert answer.status == "answered"
    assert len(answer.citations) == 1
    assert answer.citations[0].quote == CHUNKS[0].text
    assert validate_citations(answer, CHUNKS) == []


def test_build_answer_fabricated_citation_needs_review():
    answer = build_answer("依据某条【minfadian-c999】……", CHUNKS, latency_ms=100)
    assert answer.status == "needs_review"
    assert answer.citations == []  # 编造引用不进 citations


def test_build_answer_no_citation_is_refused():
    answer = build_answer("我觉得大概是三年吧。", CHUNKS, latency_ms=100)
    assert answer.status == "refused"


def test_validate_citations_strict_substring():
    answer = build_answer("期间为三年【minfadian-c003】。", CHUNKS, latency_ms=100)
    # 篡改 quote：不是原文逐字子串，严格校验必须发现
    answer.citations[0].quote = "诉讼时效期间为五年。"
    problems = validate_citations(answer, CHUNKS)
    assert any("逐字子串" in p for p in problems)


# ---- OpenAI 兼容客户端 ----


def test_openai_chat_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "期间为三年【minfadian-c003】。"}}]},
        )

    client = OpenAIChatClient(
        base_url="http://fake-llm/v1",
        model="qwen2.5:7b",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert isinstance(client, LLMClient)
    reply = client.chat([{"role": "user", "content": "诉讼时效多久？"}])
    assert "minfadian-c003" in reply
