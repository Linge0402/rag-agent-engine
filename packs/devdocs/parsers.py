"""技术文档解析与切分：官方文档 Markdown（YAML front matter + 正文）。

架构验证点：本包的切分**直接复用** core 的 StructuralChunkStrategy
（标题层级切分），包内只做一层领域 metadata 装饰——
接入第二个领域不需要为切分重写一行核心逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from core.chunking.base import ChunkStrategy
from core.chunking.structural import StructuralChunkStrategy
from core.parsing.base import ParseError
from core.service.schemas import Chunk, ParsedDocument

# front matter 读取复用法律包的公开实现；若后续出现第三个用 front matter
# 的领域包，应把它提升到 core.parsing（届时两处包逻辑都无需改动）
from packs.legal.parsers import read_front_matter

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)

REQUIRED_FRONT_MATTER = ("library", "version", "title")


class DevDocsMarkdownParser:
    """技术文档 Markdown 解析器：front matter → ParsedDocument.metadata。"""

    name = "devdocs_markdown"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".md"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ParseError(f"无法读取文件 {path}: {e}") from e

        try:
            metadata = read_front_matter(raw)
        except yaml.YAMLError as e:
            raise ParseError(f"front matter 解析失败 {path}: {e}") from e
        if metadata is None:
            raise ParseError(f"文档缺少 YAML front matter: {path}")

        missing = [k for k in REQUIRED_FRONT_MATTER if k not in metadata]
        if missing:
            raise ParseError(f"front matter 缺少必填字段 {missing}: {path}")

        body = raw[_FRONT_MATTER_RE.match(raw).end() :]
        if not body.strip():
            raise ParseError(f"文档正文为空: {path}")
        return ParsedDocument(doc_id=path.stem, source=str(path), text=body, metadata=metadata)


class DevDocsChunkStrategy(ChunkStrategy):
    """core 结构切分 + 领域 metadata 装饰（库名/版本号/文档链接）。

    装饰器模式：标题层级切分的全部逻辑复用 StructuralChunkStrategy，
    本类只负责把 front matter 里的领域 metadata 补到每个 chunk 上。
    """

    def __init__(self, max_chars: int = 800) -> None:
        self._inner = StructuralChunkStrategy(max_chars=max_chars)

    @property
    def name(self) -> str:
        return "devdocs_structural"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = self._inner.chunk(doc)
        for chunk in chunks:
            meta = chunk.metadata
            meta.source = str(doc.metadata.get("title", doc.source))
            meta.version = str(doc.metadata.get("version", "")) or None
            meta.extra.update(
                {
                    "library": doc.metadata.get("library"),
                    "version": doc.metadata.get("version"),
                    "doc_url": doc.metadata.get("doc_url", ""),
                }
            )
        return chunks
