"""解析器协议。

引擎内置 loaders.py 覆盖通用格式（md/pdf/docx/doc/excel）；
领域包可以提供专用解析器（如法规 XML、文档站 HTML），
通过 KnowledgePack.parsers() 注入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from core.service.schemas import ParsedDocument


@runtime_checkable
class Parser(Protocol):
    """把一份原始文件解析成 ParsedDocument。

    实现要求：
    - supports() 决定入库分流时该文件是否归此解析器处理；
    - 扫描件等无文本层输入应返回空 text 并在 metadata 中标记
      needs_ocr=True，由流水线决定是否走 OCR 兜底，解析器自身不做 OCR；
    - 解析失败应抛出 ParseError，而不是返回残缺结果。
    """

    @property
    def name(self) -> str: ...

    def supports(self, path: Path) -> bool: ...

    def parse(self, path: Path) -> ParsedDocument: ...


class ParseError(Exception):
    """解析失败。message 中应包含文件路径与原因。"""
