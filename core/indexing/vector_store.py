"""Qdrant 向量库封装。

三个关键设计：
- 幂等写入：point id 由 chunk_id 经 UUID5 确定性生成，重复入库同一文档
  不会产生重复数据（重复 upsert 同一 id 即覆盖）；
- payload 扁平映射：ChunkMetadata 内置字段存 "metadata.<字段>"，
  领域扩展字段存 "metadata.extra.<字段>"，过滤条件按此约定寻址；
- 过滤检索：search() 接受等值条件字典，翻译成 Qdrant Filter；
  声明式过滤 DSL（条件组合、字段合法性校验）在 W2 的 retrieval/filters.py 强化。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from core.service.schemas import Chunk, ChunkMetadata

# 固定命名空间：point id 只由 chunk_id 决定，与运行实例无关
_POINT_NAMESPACE = uuid.UUID("8f2e4a1c-7b3d-4e5f-9a6b-1c2d3e4f5a6b")

_METADATA_FIELDS = set(ChunkMetadata.model_fields)


def point_id(chunk_id: str) -> str:
    """chunk_id → 确定性 point id（幂等写入的基础）。"""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def payload_key(name: str) -> str:
    """过滤字段名 → payload 路径。内置字段与 extra 扩展字段分开寻址。"""
    if name in _METADATA_FIELDS:
        return f"metadata.{name}"
    return f"metadata.extra.{name}"


def build_filter(conditions: dict[str, Any]) -> models.Filter | None:
    """等值条件字典 → Qdrant Filter（各条件之间为 AND）。"""
    if not conditions:
        return None
    must = [
        models.FieldCondition(key=payload_key(name), match=models.MatchValue(value=value))
        for name, value in conditions.items()
    ]
    return models.Filter(must=must)


@dataclass
class ScoredChunk:
    """一次召回的命中结果：chunk + 向量相似度分数。"""

    chunk: Chunk
    score: float


class QdrantVectorStore:
    """Qdrant collection 管理 + 幂等写入 + 过滤检索。"""

    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        client: QdrantClient | None = None,
    ) -> None:
        self._client = client or QdrantClient(url=url)
        self._collection = collection
        self._dim = dim

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """collection 不存在则创建；已存在则校验维度一致（不一致说明换过模型，直接报错）。"""
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
            return
        info = self._client.get_collection(self._collection)
        existing = info.config.params.vectors.size
        if existing != self._dim:
            raise ValueError(
                f"collection {self._collection!r} 维度为 {existing}，"
                f"与当前 embedding 维度 {self._dim} 不一致；请换 collection 名或重建"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """幂等写入，返回写入条数。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata.model_dump(),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """向量召回，可附加 metadata 等值硬过滤。"""
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=build_filter(filters or {}),
            with_payload=True,
        )
        return [ScoredChunk(chunk=_to_chunk(p.payload), score=p.score) for p in response.points]

    def get(self, chunk_id: str) -> Chunk | None:
        """按 chunk_id 取原文（引用核验、MCP get_chunk 工具都走这里）。"""
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id(chunk_id)],
            with_payload=True,
        )
        if not records:
            return None
        return _to_chunk(records[0].payload)


def _to_chunk(payload: dict[str, Any] | None) -> Chunk:
    if payload is None:
        raise ValueError("point 缺少 payload")
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        metadata=ChunkMetadata.model_validate(payload["metadata"]),
    )
