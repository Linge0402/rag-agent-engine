"""Markdown 解析与入库分流测试。"""

from pathlib import Path

import pytest

from core.parsing.base import ParseError
from core.parsing.loaders import DEFAULT_PARSERS, MarkdownParser, pick_parser


def test_parse_markdown(tmp_path: Path):
    f = tmp_path / "民法典节选.md"
    f.write_text("# 第一编 总则\n\n第一百八十八条 诉讼时效期间为三年。\n")

    doc = MarkdownParser().parse(f)

    assert doc.doc_id == "民法典节选"
    assert doc.source == str(f)
    assert "第一百八十八条" in doc.text


def test_supports_markdown_extensions(tmp_path: Path):
    parser = MarkdownParser()
    assert parser.supports(tmp_path / "a.md")
    assert parser.supports(tmp_path / "b.markdown")
    assert parser.supports(tmp_path / "c.MD")  # 大小写不敏感
    assert not parser.supports(tmp_path / "d.pdf")


def test_empty_file_raises(tmp_path: Path):
    f = tmp_path / "empty.md"
    f.write_text("   \n  ")
    with pytest.raises(ParseError, match="内容为空"):
        MarkdownParser().parse(f)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ParseError, match="无法读取"):
        MarkdownParser().parse(tmp_path / "不存在.md")


def test_pick_parser_no_match(tmp_path: Path):
    with pytest.raises(ParseError, match="没有解析器支持"):
        pick_parser(tmp_path / "法律汇编.pdf", DEFAULT_PARSERS)
