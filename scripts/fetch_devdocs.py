"""技术文档采集管线：LangChain 官方文档站 → data/devdocs/*.md。

用法：
    uv run python scripts/fetch_devdocs.py

设计要点：
- docs.langchain.com 为每个页面提供 .md 版本（llms.txt 标准），
  采集的是官方维护的 Markdown，无需 HTML 清洗；
- front matter 声明 metadata：库名 + 版本号（技术文档的核心过滤维度，
  对应法律包的效力级别/时效状态）；
- 版本号取自本地安装的库版本——文档内容与代码环境对齐才有意义，
  文档站若已更新到更新版本，以重新采集时的声明为准；
- 输出幂等：同一页面重复运行产生相同文件。
"""

from __future__ import annotations

import importlib.metadata
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "devdocs"
BASE = "https://docs.langchain.com/oss/python/langgraph"

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0"

# 采集清单：LangGraph 核心概念页
PAGES = [
    "overview",
    "quickstart",
    "graph-api",
    "workflows-agents",
    "persistence",
    "streaming",
    "memory",
    "human-in-the-loop",
]


@dataclass(frozen=True)
class DocSource:
    library: str
    slug: str
    version: str


def strip_boilerplate(md: str) -> str:
    """去掉 .md 页面顶部的 llms.txt 指引块（对检索是噪声）。"""
    lines = md.splitlines()
    out: list[str] = []
    skip_quote = False
    for line in lines:
        if line.startswith("> ## Documentation Index"):
            skip_quote = True
            continue
        if skip_quote:
            if line.startswith(">") or not line.strip():
                continue
            skip_quote = False
        out.append(line)
    return "\n".join(out).strip() + "\n"


def extract_title(md: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, re.M)
    return m.group(1).strip() if m else fallback


def fetch(url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        version = importlib.metadata.version("langgraph")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    print(f"本地 langgraph 版本: {version}")

    for slug in PAGES:
        url = f"{BASE}/{slug}.md"
        body = strip_boilerplate(fetch(url))
        title = extract_title(body, slug)
        front = {
            "library": "langgraph",
            "version": version,
            "title": title,
            "doc_url": f"{BASE}/{slug}",
        }
        content = (
            "---\n"
            + yaml.safe_dump(front, allow_unicode=True, sort_keys=False)
            + "---\n\n"
            + body
        )
        target = DATA_DIR / f"langgraph-{slug}.md"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=DATA_DIR, delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(content)
        Path(tmp.name).replace(target)
        print(f"[OK] {slug}: {title} | {len(body)} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
