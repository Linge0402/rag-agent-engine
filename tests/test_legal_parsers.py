"""法律切分器与解析器测试。"""

from pathlib import Path

import pytest

from core.parsing.base import ParseError
from packs.legal.parsers import LegalChunkStrategy, LegalMarkdownParser, cn2int

# ---------------------------------------------------------------------------
# cn2int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("一", 1), ("十", 10), ("十三", 13), ("二十", 20),
        ("一百零一", 101), ("二百一十八", 218), ("一千二百六十", 1260),
    ],
)
def test_cn2int(text, expected):
    assert cn2int(text) == expected


def test_cn2int_rejects_garbage():
    with pytest.raises(ValueError):
        cn2int("第X条")


# ---------------------------------------------------------------------------
# 法规 Markdown 解析器
# ---------------------------------------------------------------------------

MINI_LAW = """---
title: 测试法
level: 法律
issuing_authority: 全国人民代表大会
validity: 现行有效
effective_date: 2024-01-01
---

第一条　为了测试，制定本法。

第二章　适用

本法第一条　的规定适用于一切测试。

第二条　本法自2024年1月1日起施行。
"""


def test_parser_extracts_front_matter(tmp_path: Path):
    path = tmp_path / "testlaw.md"
    path.write_text(MINI_LAW, encoding="utf-8")
    doc = LegalMarkdownParser().parse(path)
    assert doc.doc_id == "testlaw"
    assert doc.metadata["title"] == "测试法"
    assert doc.metadata["validity"] == "现行有效"
    assert "第一条" in doc.text and "---" not in doc.text


def test_parser_rejects_missing_front_matter(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_text("第一条　没有 front matter。", encoding="utf-8")
    with pytest.raises(ParseError, match="front matter"):
        LegalMarkdownParser().parse(path)


def test_parser_rejects_missing_required_fields(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_text("---\ntitle: 只有标题\n---\n\n第一条　正文。", encoding="utf-8")
    with pytest.raises(ParseError, match="缺少必填字段"):
        LegalMarkdownParser().parse(path)


# ---------------------------------------------------------------------------
# 切分：单调递增规则排除交叉引用
# ---------------------------------------------------------------------------


def test_chunker_excludes_inline_cross_reference(tmp_path: Path):
    """"本法第一条"被换行拆到行首时不应被当作新条文——条号未递增。"""
    doc = LegalMarkdownParser().parse(_write(tmp_path, MINI_LAW))
    chunks = LegalChunkStrategy().chunk(doc)
    assert len(chunks) == 2  # 只有第一条、第二条；交叉引用不产生第三个 chunk
    assert chunks[0].metadata.extra["article_no"] == 1
    assert chunks[1].metadata.extra["article_no"] == 2


def test_chunker_metadata_and_ids(tmp_path: Path):
    doc = LegalMarkdownParser().parse(_write(tmp_path, MINI_LAW))
    chunks = LegalChunkStrategy().chunk(doc)
    first = chunks[0]
    assert first.chunk_id == "testlaw-t0001"
    assert first.metadata.validity == "现行有效"
    assert first.metadata.extra["level"] == "法律"
    assert "第二章\u3000适用" in chunks[1].metadata.section_path


def test_chunker_tracks_hierarchy(tmp_path: Path):
    law = """---
title: 层级法
level: 法律
issuing_authority: 测试机关
validity: 现行有效
effective_date: 2024-01-01
---

第一编　总则

第一章　一般规定

第一条　甲。

第二节　特别规定

第二条　乙。

第二编　分则

第三条　丙。
"""
    chunks = LegalChunkStrategy().chunk(
        LegalMarkdownParser().parse(_write(tmp_path, law, name="cengjifa.md"))
    )
    assert chunks[0].metadata.section_path == ["第一编　总则", "第一章　一般规定"]
    assert chunks[1].metadata.section_path == ["第一编　总则", "第一章　一般规定", "第二节　特别规定"]
    assert chunks[2].metadata.section_path == ["第二编　分则"]  # 新编重置下属层级
    assert chunks[2].chunk_id == "cengjifa-t0003"


def _write(tmp_path: Path, content: str, name: str = "testlaw.md") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path
