"""法律领域解析与切分：法规"编-章-节-条"层级结构。

与通用 structural 切分的本质区别：法规的最小语义单元是"条"，
而"条"的归属路径（编/分编/章/节）与效力 metadata 是检索质量的关键——
回答"诉讼时效多久"时，命中【民法典·第一编 总则·第九章 诉讼时效】
第一百八十八条，比命中一段无出处的文本可信得多。

切分规则：
- 条号识别要求**单调递增**：法规正文条号从第一条到第 N 条严格递增，
  行内交叉引用（如"本法第四十九条的规..."被网页换行拆断后形如行首条号）
  因条号小于等于当前条号而被排除——这是区分"条文起始"与"引用"的
  确定性判据，不依赖脆弱的格式假设；
- 每个"条"一个 chunk，正文以层级面包屑开头（供 embedding 与引用展示）；
- chunk_id 确定性：{doc_id}-t{条号:04d}（幂等入库依赖）；
- metadata：编/章/节入 section_path；效力级别/时效状态/发文机关/施行日期
  来自文件 front matter（由 scripts/fetch_legal.py 采集时声明）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from core.chunking.base import ChunkStrategy
from core.parsing.base import ParseError
from core.service.schemas import Chunk, ChunkMetadata, ParsedDocument

# ---------------------------------------------------------------------------
# 中文数字解析（法规条号最大到一千二百六十，覆盖万以内即可）
# ---------------------------------------------------------------------------

_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn2int(text: str) -> int:
    """中文数字 → 整数。支持"十/百/千"组合（如 一千二百六十、一百零一、十三）。"""
    if text.isdigit():
        return int(text)
    total, section = 0, 0
    for ch in text:
        if ch in _DIGITS:
            section = _DIGITS[ch]
        elif ch in _UNITS:
            unit = _UNITS[ch]
            section = unit if section == 0 else section * unit  # "十三" = 10+3 的 "十"
            total += section
            section = 0
        else:
            raise ValueError(f"无法解析的中文数字: {text!r}")
    return total + section


# ---------------------------------------------------------------------------
# 法规结构标记
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百千零]+)条[　\s]*(.*)$")
_BIAN_RE = re.compile(r"^第[一二三四五六七八九十]+编[　\s]+(.+)$")
_FENBIAN_RE = re.compile(r"^第[一二三四五六七八九十]+分编[　\s]+(.+)$")
_ZHANG_RE = re.compile(r"^第[一二三四五六七八九十百千]+章[　\s]+(.+)$")
_JIE_RE = re.compile(r"^第[一二三四五六七八九十]+节[　\s]+(.+)$")
_FUZE_RE = re.compile(r"^附[　\s]*则$")

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)

REQUIRED_FRONT_MATTER = ("title", "level", "issuing_authority", "validity", "effective_date")


def read_front_matter(raw: str) -> dict[str, Any] | None:
    """读取法规文件的 YAML front matter；无 front matter 返回 None。

    解析器与登记簿（registry.py）共用的唯一入口，保证两处口径一致。
    """
    match = _FRONT_MATTER_RE.match(raw)
    if match is None:
        return None
    return yaml.safe_load(match.group(1)) or {}


# ---------------------------------------------------------------------------
# 解析器：带 YAML front matter 的法规 Markdown
# ---------------------------------------------------------------------------


class LegalMarkdownParser:
    """法规 Markdown 解析器：front matter → ParsedDocument.metadata。

    front matter 是效力级别/时效状态等关键 metadata 的唯一来源，
    缺失必填字段直接报错——metadata 不完整的法规不允许入库。
    """

    name = "legal_markdown"

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
            raise ParseError(f"法规文件缺少 YAML front matter: {path}")

        missing = [k for k in REQUIRED_FRONT_MATTER if k not in metadata]
        if missing:
            raise ParseError(f"法规 front matter 缺少必填字段 {missing}: {path}")

        body = raw[_FRONT_MATTER_RE.match(raw).end() :]
        if not body.strip():
            raise ParseError(f"法规正文为空: {path}")
        return ParsedDocument(doc_id=path.stem, source=str(path), text=body, metadata=metadata)


# ---------------------------------------------------------------------------
# 切分策略：编-分编-章-节-条
# ---------------------------------------------------------------------------


class LegalChunkStrategy(ChunkStrategy):
    """按"条"切分法规，层级路径入 section_path。"""

    @property
    def name(self) -> str:
        return "legal_articles"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        bian, fenbian, zhang, jie = "", "", "", ""
        last_article_no = 0
        article_lines: list[str] = []

        def section_path() -> list[str]:
            return [p for p in (bian, fenbian, zhang, jie) if p]

        def flush_article() -> None:
            if not article_lines:
                return
            header = article_lines[0]
            body = "\n".join(article_lines).strip()
            breadcrumb = "·".join([doc.metadata.get("title", doc.doc_id), *section_path()])
            no = last_article_no
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-t{no:04d}",
                    doc_id=doc.doc_id,
                    text=f"【{breadcrumb}】\n{body}",
                    metadata=ChunkMetadata(
                        source=doc.metadata.get("title", doc.source),
                        section_path=section_path(),
                        version=str(doc.metadata.get("effective_date", "")) or None,
                        validity=doc.metadata.get("validity"),
                        extra={
                            "law_id": doc.doc_id,
                            "level": doc.metadata.get("level"),
                            "issuing_authority": doc.metadata.get("issuing_authority"),
                            "effective_date": str(doc.metadata.get("effective_date", "")),
                            "article_no": no,
                        },
                    ),
                )
            )
            article_lines.clear()
            _ = header  # 保留变量语义清晰：header 已并入 body

        for line in doc.text.splitlines():
            stripped = line.strip()
            if not stripped:
                if article_lines and article_lines[-1] != "":
                    article_lines.append("")
                continue
            if m := _BIAN_RE.match(stripped):
                flush_article()
                bian, fenbian, zhang, jie = m.group(0), "", "", ""
                continue
            if m := _FENBIAN_RE.match(stripped):
                flush_article()
                fenbian, zhang, jie = m.group(0), "", ""
                continue
            if m := _ZHANG_RE.match(stripped):
                flush_article()
                zhang, jie = m.group(0), ""
                continue
            if m := _JIE_RE.match(stripped):
                flush_article()
                jie = m.group(0)
                continue
            if _FUZE_RE.match(stripped):
                flush_article()
                bian, fenbian, zhang, jie = stripped, "", "", ""
                continue
            if m := _ARTICLE_RE.match(stripped):
                no = cn2int(m.group(1))
                if no > last_article_no:
                    # 条号单调递增 → 确为新条文起始（交叉引用条号必小于当前值）
                    flush_article()
                    last_article_no = no
                    article_lines.append(stripped)
                    continue
            if article_lines:
                article_lines.append(stripped)
            # 正文开始前的目录/ preamble 不产出 chunk（无条号归属）

        flush_article()
        return chunks
