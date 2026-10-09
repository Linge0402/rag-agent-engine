"""Embedding 客户端。

`EmbeddingClient` 是协议：入库与查询两侧都只依赖协议，不关心向量从哪来。
`OpenAIEmbeddingClient` 走 OpenAI 兼容的 /embeddings 端点（bge-m3、Ollama 等均可）。
测试中注入 Fake 实现，保证离线可跑。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from core.config import Settings


@runtime_checkable
class EmbeddingClient(Protocol):
    """文本向量化。

    实现要求：
    - embed() 返回的向量顺序必须与输入文本顺序一致；
    - 同一模型下 dim 恒定。
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    """OpenAI 兼容端点的 embedding 客户端，支持分批请求。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._dim: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> OpenAIEmbeddingClient:
        return cls(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        """向量维度，首次访问时用一条探测文本确定（避免硬编码模型维度）。"""
        if self._dim is None:
            self._dim = len(self.embed(["维度探测"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": batch},
                headers=self._headers,
            )
            resp.raise_for_status()
            # 按返回中的 index 字段对齐输入顺序，不信任返回顺序
            items = sorted(resp.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in items)
        return vectors
