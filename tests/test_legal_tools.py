"""法律领域工具测试（离线：tmp 目录登记簿 + Fake store）。"""

from pathlib import Path

import pytest

from core.agent.tools import QueryContext
from core.service.schemas import Chunk, ChunkMetadata
from packs.legal.registry import LawRegistry
from packs.legal.tools import (
    CheckValidityArgs,
    CheckValidityTool,
    CompareVersionsArgs,
    CompareVersionsTool,
    LocateArticleArgs,
    LocateArticleTool,
)

OLD_LAW = """---
title: 示例公司法（2018年修正）
level: 法律
issuing_authority: 全国人民代表大会常务委员会
validity: 已废止
effective_date: 2018-10-26
---

第一条　旧版第一条。

第二条　两版相同的规定。
"""

NEW_LAW = """---
title: 示例公司法（2023年修订）
level: 法律
issuing_authority: 全国人民代表大会常务委员会
validity: 现行有效
effective_date: 2024-07-01
---

第一条　新版第一条，内容已修改。

第二条　两版相同的规定。

第三条　新法新增的条款。
"""


@pytest.fixture
def registry(tmp_path: Path) -> LawRegistry:
    (tmp_path / "gsf-2018.md").write_text(OLD_LAW, encoding="utf-8")
    (tmp_path / "gsf-2023.md").write_text(NEW_LAW, encoding="utf-8")
    return LawRegistry(tmp_path)


@pytest.fixture
def ctx() -> QueryContext:
    return QueryContext(session_id="t", pack_name="legal")


class FakeStore:
    def __init__(self, chunks: dict[str, Chunk]):
        self._chunks = chunks

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


async def test_check_validity_finds_both_versions(registry, ctx):
    result = await CheckValidityTool(registry).run(CheckValidityArgs(law_name="公司法"), ctx)
    assert result.ok
    titles = {r["title"] for r in result.data["results"]}
    assert titles == {"示例公司法（2018年修正）", "示例公司法（2023年修订）"}


async def test_check_validity_unknown_law(registry, ctx):
    result = await CheckValidityTool(registry).run(CheckValidityArgs(law_name="刑法"), ctx)
    assert not result.ok
    assert "未收录" in result.error


async def test_locate_article_uses_deterministic_id(registry, ctx):
    chunk = Chunk(
        chunk_id="gsf-2023-t0002",
        doc_id="gsf-2023",
        text="第二条　两版相同的规定。",
        metadata=ChunkMetadata(source="示例公司法（2023年修订）", validity="现行有效"),
    )
    store = FakeStore({chunk.chunk_id: chunk})
    tool = LocateArticleTool(registry, store)
    result = await tool.run(LocateArticleArgs(law_name="公司法", article_no=2), ctx)
    assert result.ok
    assert result.chunks[0].text.startswith("第二条")
    assert "2018" in result.data["note"]  # 提醒存在旧版本
    assert result.data["validity"] == "现行有效"


async def test_locate_article_missing(registry, ctx):
    store = FakeStore({})
    result = await LocateArticleTool(registry, store).run(
        LocateArticleArgs(law_name="公司法", article_no=999), ctx
    )
    assert not result.ok
    assert "999" in result.error


async def test_compare_versions(registry, ctx):
    result = await CompareVersionsTool(registry).run(
        CompareVersionsArgs(law_name="公司法", article_no=1), ctx
    )
    assert result.ok
    assert result.data["old"]["law_id"] == "gsf-2018"
    assert result.data["new"]["law_id"] == "gsf-2023"
    assert result.data["old_article_count"] == 2
    assert result.data["new_article_count"] == 3
    assert result.data["only_in_new"] == [3]
    assert result.data["same_no_changed_count"] == 1  # 第一条变了，第二条相同
    assert result.data["focus"]["changed"] is True
    assert "新版" in result.data["focus"]["new_text"]


async def test_compare_versions_single_version(registry: LawRegistry, tmp_path: Path, ctx):
    single = LawRegistry(tmp_path / "empty")  # 目录不存在 → 空登记簿
    result = await CompareVersionsTool(single).run(CompareVersionsArgs(law_name="公司法"), ctx)
    assert not result.ok
