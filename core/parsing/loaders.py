"""多格式文档解析入口。

W1b 阶段仅实现 Markdown 解析；pdf/docx/doc/excel 与 OCR 兜底在 W2 补齐
（安全限制：ZIP entry 数、解压大小、图片数上限，见 docs/implementation-plan.md）。
"""

from __future__ import annotations

from pathlib import Path

from core.parsing.base import ParseError, Parser
from core.service.schemas import ParsedDocument


class MarkdownParser:
    """Markdown 文件解析器。"""

    name = "markdown"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def parse(self, path: Path) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ParseError(f"无法读取文件 {path}: {e}") from e
        if not text.strip():
            raise ParseError(f"文件内容为空: {path}")
        return ParsedDocument(doc_id=path.stem, source=str(path), text=text)


DEFAULT_PARSERS: list[Parser] = [MarkdownParser()]


def pick_parser(path: Path, parsers: list[Parser]) -> Parser:
    """入库分流：按声明顺序找到第一个支持该文件的解析器。"""
    for parser in parsers:
        if parser.supports(path):
            return parser
    raise ParseError(f"没有解析器支持该文件: {path}")
