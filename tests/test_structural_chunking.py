"""结构切分测试：标题层级、确定性、不丢内容。

示例文本使用公开法规语料（民法典结构为公开信息）。
"""

from core.chunking.structural import StructuralChunkStrategy
from core.service.schemas import ParsedDocument

DOC_TEXT = """# 第一编 总则

## 第一章 基本规定

第一条 为了保护民事主体的合法权益，调整民事关系，制定本法。

## 第二章 自然人

第十三条 自然人从出生时起到死亡时止，具有民事权利能力。

# 第二编 物权

第二百零五条 本编调整因物的归属和利用产生的民事关系。
"""


def make_doc(text: str = DOC_TEXT) -> ParsedDocument:
    return ParsedDocument(doc_id="minfadian", source="民法典.md", text=text)


def test_section_path_follows_heading_hierarchy():
    chunks = StructuralChunkStrategy().chunk(make_doc())
    paths = [c.metadata.section_path for c in chunks]

    assert ["第一编 总则", "第一章 基本规定"] in paths
    assert ["第一编 总则", "第二章 自然人"] in paths
    assert ["第二编 物权"] in paths  # 新的一级标题重置层级


def test_heading_merged_into_chunk_text():
    chunks = StructuralChunkStrategy().chunk(make_doc())
    first = chunks[0]
    assert first.text.startswith("第一章 基本规定")
    assert "第一条" in first.text


def test_chunk_ids_are_deterministic():
    strategy = StructuralChunkStrategy()
    ids_1 = [c.chunk_id for c in strategy.chunk(make_doc())]
    ids_2 = [c.chunk_id for c in strategy.chunk(make_doc())]
    assert ids_1 == ids_2
    assert ids_1[0] == "minfadian-c001"


def test_long_section_splits_by_paragraph():
    body = "\n\n".join(f"第{i}条 " + "内容" * 50 for i in range(1, 10))
    doc = make_doc(f"# 编\n\n{body}")

    chunks = StructuralChunkStrategy().chunk(doc)

    assert len(chunks) > 1
    # 不丢内容：所有条款都出现在某个 chunk 中
    merged = "\n".join(c.text for c in chunks)
    for i in range(1, 10):
        assert f"第{i}条" in merged


def test_content_before_first_heading():
    doc = make_doc("前言：本法于2020年通过。\n\n# 第一编 总则\n\n第一条 ……")
    chunks = StructuralChunkStrategy().chunk(doc)
    assert chunks[0].metadata.section_path == []
    assert "前言" in chunks[0].text


def test_invalid_max_chars():
    import pytest

    with pytest.raises(ValueError):
        StructuralChunkStrategy(max_chars=0)
