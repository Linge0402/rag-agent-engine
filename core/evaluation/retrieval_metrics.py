"""检索质量指标：top-k 命中率与 MRR。

输入是一次检索的命中序列与期望命中的 chunk_id 集合，
输出是该样本的指标；汇总由调用方（评估脚本）完成。
指标口径遵循信息检索惯例，便于与公开基线对齐表述。
"""

from __future__ import annotations


def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """top-k 内是否命中任一期望 chunk：命中 1.0，否则 0.0。"""
    if not expected_ids:
        raise ValueError("expected_ids 为空：该样本无法用于检索断言")
    top = retrieved_ids[:k]
    return 1.0 if any(e in top for e in expected_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """倒数排名：第一个期望 chunk 的名次倒数（第 1 名=1.0，第 2 名=0.5……），未命中为 0。"""
    if not expected_ids:
        raise ValueError("expected_ids 为空：该样本无法用于检索断言")
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            return 1.0 / rank
    return 0.0
