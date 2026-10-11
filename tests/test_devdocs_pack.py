"""技术文档领域包测试：解析/切分装饰、工具、pack 自检。"""

from pathlib import Path

import pytest

from core.agent.tools import QueryContext
from core.pack import validate_pack
from core.parsing.base import ParseError
from packs.devdocs import DevDocsPack
from packs.devdocs.parsers import DevDocsChunkStrategy, DevDocsMarkdownParser
from packs.devdocs.tools import RunPythonArgs, RunPythonTool, VersionCompatArgs, VersionCompatTool

MINI_DOC = """---
library: langgraph
version: 1.2.14
title: Graph API overview
doc_url: https://example.com/graph-api
---

# Graph API overview

## StateGraph

StateGraph 是核心类。

```python
from langgraph.graph import StateGraph
```

### add_node

add_node 方法添加节点。
"""


@pytest.fixture
def ctx() -> QueryContext:
    return QueryContext(session_id="t", pack_name="devdocs")


# ---------------------------------------------------------------------------
# 解析 + 切分装饰
# ---------------------------------------------------------------------------


def test_parser_extracts_front_matter(tmp_path: Path):
    path = tmp_path / "langgraph-x.md"
    path.write_text(MINI_DOC, encoding="utf-8")
    doc = DevDocsMarkdownParser().parse(path)
    assert doc.metadata["library"] == "langgraph"
    assert doc.metadata["version"] == "1.2.14"


def test_parser_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_text("---\ntitle: 只有标题\n---\n\n正文。", encoding="utf-8")
    with pytest.raises(ParseError, match="缺少必填字段"):
        DevDocsMarkdownParser().parse(path)


def test_chunk_strategy_decorates_metadata(tmp_path: Path):
    """切分逻辑复用 core structural，领域 metadata 由装饰器补齐。"""
    path = tmp_path / "langgraph-x.md"
    path.write_text(MINI_DOC, encoding="utf-8")
    doc = DevDocsMarkdownParser().parse(path)
    chunks = DevDocsChunkStrategy().chunk(doc)
    assert chunks  # 结构切分生效
    for c in chunks:
        assert c.metadata.version == "1.2.14"
        assert c.metadata.extra["library"] == "langgraph"
        assert c.metadata.source == "Graph API overview"


# ---------------------------------------------------------------------------
# 版本兼容工具
# ---------------------------------------------------------------------------


async def test_version_compat_hit(ctx):
    result = await VersionCompatTool().run(
        VersionCompatArgs(library="langgraph", version="1.2.14"), ctx
    )
    assert result.ok
    assert "langchain-core" in result.data["dependencies"]


async def test_version_compat_unknown_version(ctx):
    result = await VersionCompatTool().run(
        VersionCompatArgs(library="langgraph", version="0.0.1"), ctx
    )
    assert not result.ok
    assert "已知版本" in result.error


async def test_version_compat_unknown_library(ctx):
    result = await VersionCompatTool().run(VersionCompatArgs(library="pytorch", version=""), ctx)
    assert not result.ok


# ---------------------------------------------------------------------------
# 代码沙箱工具
# ---------------------------------------------------------------------------


async def test_run_python_executes_code(ctx):
    result = await RunPythonTool().run(RunPythonArgs(code="print(1 + 1)"), ctx)
    assert result.ok
    assert result.data["returncode"] == 0
    assert "2" in result.data["output"]


async def test_run_python_captures_error(ctx):
    result = await RunPythonTool().run(RunPythonArgs(code="1/0"), ctx)
    assert result.ok  # 工具本身成功，是被执行代码出错
    assert result.data["returncode"] != 0
    assert "ZeroDivisionError" in result.data["output"]


async def test_run_python_blocks_dangerous_pattern(ctx):
    result = await RunPythonTool().run(RunPythonArgs(code="import os; os.system('ls')"), ctx)
    assert not result.ok
    assert "拦截" in result.error


async def test_run_python_timeout(ctx):
    result = await RunPythonTool().run(
        RunPythonArgs(code="import time; time.sleep(60)"), ctx
    )
    assert not result.ok
    assert "超时" in result.error


# ---------------------------------------------------------------------------
# pack 自检
# ---------------------------------------------------------------------------


def test_devdocs_pack_passes_validation():
    assert validate_pack(DevDocsPack()) == []
