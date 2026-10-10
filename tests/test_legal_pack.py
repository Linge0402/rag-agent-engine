"""LegalPack 自检 + 评估指标测试。"""


from core.evaluation.answer_metrics import citation_accuracy, is_correct_refusal
from core.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank
from core.pack import validate_pack
from core.service.schemas import Answer, Chunk, ChunkMetadata, Citation
from packs.legal import LegalPack


def test_legal_pack_passes_validation():
    """六个声明齐全、评估集非空——新领域接入的准入门槛。"""
    problems = validate_pack(LegalPack())
    assert problems == []


def test_legal_pack_eval_set_covers_three_categories():
    eval_set = LegalPack().eval_sets()[0]
    assert len(eval_set.cases) >= 6
    has_retrieval = any(c.expected_chunk_ids for c in eval_set.cases)
    has_refusal = any(c.should_refuse for c in eval_set.cases)
    assert has_retrieval and has_refusal


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["b"], 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["x"]) == 0.0
    assert reciprocal_rank(["a"], ["a"]) == 1.0


# ---------------------------------------------------------------------------
# 生成指标
# ---------------------------------------------------------------------------

CHUNK = Chunk(
    chunk_id="c1",
    doc_id="d",
    text="诉讼时效期间为三年。",
    metadata=ChunkMetadata(source="民法典"),
)


def _answer(citations: list[Citation], status: str = "answered") -> Answer:
    return Answer(text="...", citations=citations, confidence=0.8, status=status, latency_ms=1)


def test_citation_accuracy_all_good():
    ans = _answer([Citation(chunk_id="c1", quote="诉讼时效期间为三年。")])
    assert citation_accuracy(ans, [CHUNK]) == 1.0


def test_citation_accuracy_fabricated_and_bad_quote():
    ans = _answer(
        [
            Citation(chunk_id="ghost", quote="不存在的条文"),
            Citation(chunk_id="c1", quote="改写过的引文"),
        ]
    )
    assert citation_accuracy(ans, [CHUNK]) == 0.0


def test_refusal_direction():
    assert is_correct_refusal(_answer([], status="refused"), should_refuse=True)
    assert not is_correct_refusal(_answer([], status="refused"), should_refuse=False)
    assert is_correct_refusal(
        _answer([Citation(chunk_id="c1", quote="...")]), should_refuse=False
    )
