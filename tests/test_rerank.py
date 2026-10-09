"""重排测试：词项重合度、组合打分、排序。"""

import pytest

from core.indexing.vector_store import ScoredChunk
from core.retrieval.rerank import overlap_score, rerank, tokenize
from core.service.schemas import Chunk, ChunkMetadata


def make_scored(chunk_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            text=text,
            metadata=ChunkMetadata(source="test"),
        ),
        score=score,
    )


def test_tokenize_mixed_language():
    tokens = tokenize("StateGraph 节点怎么定义")
    assert "stategraph" in tokens
    assert "节点" in tokens  # 中文二元组
    assert "定义" in tokens


def test_overlap_score_coverage():
    assert overlap_score("诉讼时效", "诉讼时效期间为三年") == 1.0
    assert overlap_score("诉讼时效", "完全无关的内容") == 0.0
    assert overlap_score("", "任何文本") == 0.0


def test_rerank_keyword_hit_beats_higher_vector_score():
    """向量分略低但关键词命中的候选，应借重合度反超。"""
    keyword_hit = make_scored("c1", "第一百八十八条 诉讼时效期间为三年", score=0.70)
    semantic_only = make_scored("c2", "权利的保护有时间限制", score=0.75)

    result = rerank("诉讼时效是多久", [semantic_only, keyword_hit], top_k=2, alpha=0.5)

    assert result[0].chunk.chunk_id == "c1"


def test_rerank_alpha_one_keeps_vector_order():
    c1 = make_scored("c1", "高向量分但无关键词", score=0.9)
    c2 = make_scored("c2", "低向量分但命中关键词", score=0.5)
    result = rerank("关键词", [c2, c1], top_k=2, alpha=1.0)
    assert result[0].chunk.chunk_id == "c1"


def test_rerank_top_k_truncates():
    candidates = [make_scored(f"c{i}", "文本", score=0.1 * i) for i in range(10)]
    assert len(rerank("文本", candidates, top_k=3)) == 3


def test_rerank_invalid_alpha():
    with pytest.raises(ValueError):
        rerank("q", [], top_k=1, alpha=1.5)
