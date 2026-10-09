"""轻量重排：向量相似度 + 词项重合度组合打分。

向量分衡量语义相似，词项重合度兜住"关键词必须命中"的场景
（法条编号、版本号、专有名词等 embedding 容易模糊的场景）。
final_score = alpha * 向量分 + (1 - alpha) * 词项重合度
"""

from __future__ import annotations

import re

from core.indexing.vector_store import ScoredChunk

_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> set[str]:
    """中英混合分词：英文按单词（小写），中文按连续汉字段的二元组（单字独立成词）。"""
    tokens: set[str] = set()
    tokens.update(_ASCII_WORD_RE.findall(text.lower()))
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def overlap_score(query: str, text: str) -> float:
    """query 词项在 text 中的覆盖率，0.0 ~ 1.0。"""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = tokenize(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int,
    alpha: float = 0.7,
) -> list[ScoredChunk]:
    """组合打分后重排，返回 top_k。alpha 为向量分权重。"""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 须在 [0, 1] 区间内")
    scored = [
        ScoredChunk(
            chunk=c.chunk,
            score=alpha * c.score + (1 - alpha) * overlap_score(query, c.chunk.text),
        )
        for c in candidates
    ]
    return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]
