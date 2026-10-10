"""LLM 客户端与引用校验。

`LLMClient` 是协议：生成侧只依赖协议，OpenAI 兼容端点（vLLM/Ollama/云服务）
均可接入；测试注入 Fake。

引用规则（严格模式，已确认的设计决策）：
- LLM 用【chunk_id】标注出处；
- 标注了不存在于召回结果中的 chunk_id → 视为编造引用，status=needs_review；
- 每条 citation 的 quote 必须是被引用 chunk 正文的逐字子串，
  由 validate_citations() 独立校验。
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import httpx

from core.config import Settings
from core.service.schemas import Answer, Chunk, Citation

# 提取【...】标注：chunk_id 允许任意非"】"字符（含中文 doc_id）；
# 合法性不依赖正则，由 build_answer 的召回集合成员校验兜底（严格模式）。
_CITATION_RE = re.compile(r"【([^】]+)】")


@runtime_checkable
class LLMClient(Protocol):
    """chat 补全。messages 为 OpenAI chat 格式，返回助手文本。

    temperature 为单次调用级覆盖：None 时用客户端默认。
    用法约定：法律生成 0.0（稳定可核验），闲聊 0.7（自然多变）。
    """

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str: ...


class OpenAIChatClient:
    """OpenAI 兼容端点的 chat 客户端。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=120.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> OpenAIChatClient:
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            **kwargs,
        )

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature if temperature is None else temperature,
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def extract_cited_ids(text: str) -> list[str]:
    """按出现顺序提取【chunk_id】标注，去重。

    容错：小模型常把标注写成【chunk_id：xxx】——剥掉 "chunk_id" 前缀再匹配。
    标注了不存在于召回集合的 id 仍视为编造（严格性不受影响）。
    """
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(text):
        cid = re.sub(r"^chunk_id\s*[:：]\s*", "", match.group(1).strip())
        if cid:
            seen.setdefault(cid)
    return list(seen)


def strip_citation_markers(text: str) -> str:
    """剥除文本中的全部【...】引用标注（用于无 chunk 可核验的工具数据回答）。"""
    return _CITATION_RE.sub("", text)


def build_answer(
    text: str,
    retrieved: list[Chunk],
    latency_ms: int,
) -> Answer:
    """把 LLM 输出组装成 Answer，并按引用合法性定状态。

    - 有合法引用 → answered；
    - 无任何引用 → refused（无依据的回答不可信）；
    - 含编造引用（不在召回集合中的 chunk_id）→ needs_review。
    confidence 为占位启发式，W2 评估模块落地后改为真实置信度。
    """
    retrieved_map = {c.chunk_id: c for c in retrieved}
    cited_ids = extract_cited_ids(text)
    has_fabricated = any(cid not in retrieved_map for cid in cited_ids)
    citations = [
        Citation(
            chunk_id=cid,
            quote=retrieved_map[cid].text,
            source=retrieved_map[cid].metadata.source,
        )
        for cid in cited_ids
        if cid in retrieved_map
    ]

    if has_fabricated:
        status, confidence = "needs_review", 0.3
    elif citations:
        status = "answered"
        confidence = min(0.95, 0.6 + 0.1 * len(citations))
    else:
        status, confidence = "refused", 0.2

    return Answer(
        text=text,
        citations=citations,
        confidence=confidence,
        status=status,
        latency_ms=latency_ms,
    )


def validate_citations(answer: Answer, chunks: list[Chunk]) -> list[str]:
    """严格校验：每条 quote 必须是被引用 chunk 正文的逐字子串。返回问题列表。"""
    chunk_map = {c.chunk_id: c for c in chunks}
    problems: list[str] = []
    for citation in answer.citations:
        chunk = chunk_map.get(citation.chunk_id)
        if chunk is None:
            problems.append(f"引用了不存在的 chunk: {citation.chunk_id}")
        elif citation.quote not in chunk.text:
            problems.append(f"quote 不是 {citation.chunk_id} 正文的逐字子串")
    return problems
