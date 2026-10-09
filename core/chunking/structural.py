"""默认结构切分策略：按 Markdown 标题层级切分，过长章节按段落二次切分。

切分结果满足 ChunkStrategy 的三条硬性要求：
- 确定性：同一文档产生相同的 chunk_id 序列（{doc_id}-cNNN，按文档顺序编号）；
- metadata 完整性：每个 chunk 带 source 与 section_path（标题层级路径）；
- 不丢内容：标题行会并入其章节正文开头，正文按段落边界切分。
"""

from __future__ import annotations

import re

from core.chunking.base import ChunkStrategy
from core.service.schemas import Chunk, ChunkMetadata, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


class StructuralChunkStrategy(ChunkStrategy):
    """标题/段落结构切分。

    max_chars: 单个 chunk 的正文上限，超过则按段落（空行分隔）二次切分；
    单个段落自身超过上限时不再硬切（保持语义完整），由调用方约束输入规模。
    """

    def __init__(self, max_chars: int = 800) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正数")
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "structural"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_path, body in _split_sections(doc.text):
            for piece in self._pack_paragraphs(body):
                idx = len(chunks) + 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}-c{idx:03d}",
                        doc_id=doc.doc_id,
                        text=piece,
                        metadata=ChunkMetadata(
                            source=doc.source,
                            section_path=list(section_path),
                        ),
                    )
                )
        return chunks

    def _pack_paragraphs(self, body: str) -> list[str]:
        """按段落打包：尽量装满 max_chars，但不拆散单个段落。"""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            if current and current_len + len(para) > self._max_chars:
                pieces.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)
        if current:
            pieces.append("\n\n".join(current))
        return pieces


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """按标题层级切分，返回 (section_path, 正文) 序列。

    标题行并入其章节正文开头；第一个标题之前的内容归入空路径章节。
    """
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []  # (标题级别, 标题文本)
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        # 只有标题、没有实际内容的章节（如"第一编"标题下直接跟子标题）不产生 chunk
        if body and not (heading_stack and body == heading_stack[-1][1]):
            sections.append(([t for _, t in heading_stack], body))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            buffer.append(title)  # 标题并入正文，保留语义
        else:
            buffer.append(line)
    flush()
    return sections
