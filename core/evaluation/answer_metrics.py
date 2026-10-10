"""生成质量指标：引用准确率与拒答正确率。

- 引用准确率：答案的每条引用逐一核对——chunk_id 属于召回集合、
  且 quote 是该 chunk 正文的逐字子串，两者缺一即为坏引用（编造/失真）；
- 拒答正确率：该拒答的样本确实拒答、该回答的样本确实带引用回答，
  两个方向都算分——"见题就拒"和"无依据硬答"同样不合格。
"""

from __future__ import annotations

from core.service.schemas import Answer, Chunk


def citation_accuracy(answer: Answer, retrieved: list[Chunk]) -> float:
    """单条答案的引用准确率：无引用计 0（answered 必须有依据）。"""
    if not answer.citations:
        return 0.0
    retrieved_map = {c.chunk_id: c for c in retrieved}
    bad = sum(
        1
        for c in answer.citations
        if c.chunk_id not in retrieved_map or c.quote not in retrieved_map[c.chunk_id].text
    )
    return (len(answer.citations) - bad) / len(answer.citations)


def is_correct_refusal(answer: Answer, should_refuse: bool) -> bool:
    """拒答方向判定：该拒的拒了、该答的答了，都算对。"""
    refused = answer.status == "refused" or not answer.citations
    return refused == should_refuse
