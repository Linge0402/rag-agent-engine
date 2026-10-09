"""入库编排：分流 → 解析 → 切分 → 向量化 → 幂等写入。

单文档失败不中断整体入库，错误记入 IngestReport 供排查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.chunking.base import ChunkStrategy
from core.indexing.embeddings import EmbeddingClient
from core.indexing.vector_store import QdrantVectorStore
from core.parsing.base import ParseError, Parser
from core.parsing.loaders import pick_parser


@dataclass
class IngestReport:
    """一次入库的结果统计。"""

    docs: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def ingest_paths(
    paths: list[Path],
    parsers: list[Parser],
    strategy: ChunkStrategy,
    embedder: EmbeddingClient,
    store: QdrantVectorStore,
) -> IngestReport:
    """把一批文件入库。解析/切分失败的文件记入 errors，其余照常入库。"""
    report = IngestReport()
    for path in paths:
        try:
            parser = pick_parser(path, parsers)
            doc = parser.parse(path)
            chunks = strategy.chunk(doc)
            vectors = embedder.embed([c.text for c in chunks])
            written = store.upsert(chunks, vectors)
        except ParseError as e:
            report.errors.append(str(e))
            continue
        report.docs += 1
        report.chunks += written
    return report
