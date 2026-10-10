"""法规数据采集管线：官方公开网页 → data/legal/*.md（YAML front matter + 正文）。

用法：
    uv run python scripts/fetch_legal.py

设计要点：
- 数据源只收官方公开渠道（司法部、省级政府部门网站等发布的法规全文）；
- 每部法规在 LAWS 清单中声明完整 metadata（效力级别/时效状态/发文机关/
  施行日期），采集脚本只负责抽取正文，不从网页猜测时效——时效判断
  以国家法律法规数据库（flk.npc.gov.cn）的人工核验为准；
- 输出幂等：同一法规重复运行产生相同文件（正文做确定性清洗）；
- 网络失败不覆盖已有文件（先下完再写，写用临时文件 + 原子替换）。
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "legal"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass(frozen=True)
class LawSource:
    """一部法规的采集声明。metadata 字段即入库后硬过滤的依据。"""

    law_id: str  # 文件名与 doc_id 前缀，如 minfadian
    title: str
    url: str
    container_class: str  # 正文所在 div 的 class 标记
    level: str  # 效力级别：法律 / 行政法规 / 部门规章
    issuing_authority: str  # 发文机关
    validity: str  # 时效状态：现行有效 / 已修订 / 已废止
    effective_date: str  # 施行日期 YYYY-MM-DD


LAWS: list[LawSource] = [
    LawSource(
        law_id="minfadian",
        title="中华人民共和国民法典",
        url="https://www.moj.gov.cn/pub/sfbgw/zwgkztzl/2025nianzhuanti/2025mfdxcy/2025mfdxcy_mfdql/202505/t20250507_518708.html",
        container_class="TRS_Editor",
        level="法律",
        issuing_authority="全国人民代表大会",
        validity="现行有效",
        effective_date="2021-01-01",
    ),
    LawSource(
        law_id="gongsifa-2023",
        title="中华人民共和国公司法（2023年修订）",
        url="https://guangdong.chinatax.gov.cn/gdsw/zjfg/2025-12/31/content_1203d9ef77944845aeca8ce79160292d.shtml",
        container_class="ucapcontent",
        level="法律",
        issuing_authority="全国人民代表大会常务委员会",
        validity="现行有效",
        effective_date="2024-07-01",
    ),
    LawSource(
        law_id="gongsifa-2018",
        title="中华人民共和国公司法（2018年修正）",
        url="https://scjgj.beijing.gov.cn/cxfw/flfgcxfw/qyjgl/202006/t20200621_1929594.html",
        container_class="trs_word",
        level="法律",
        issuing_authority="全国人民代表大会常务委员会",
        validity="已废止",  # 2024-07-01 起由 2023 修订版替代
        effective_date="2018-10-26",
    ),
]


def extract_body(html: str, container_class: str) -> str:
    """从网页中抽取正文容器，转成清洗后的纯文本。"""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find(class_=re.compile(rf"(?:^|\s){re.escape(container_class)}(?:\s|$)"))
    if container is None:  # 有的站点用自定义标签名（如 <ucapcontent>）而非 class
        container = soup.find(container_class)
    if container is None:
        raise ValueError(f"未找到正文容器 class/tag={container_class!r}")
    return normalize(container.get_text("\n"))


def normalize(text: str) -> str:
    """确定性清洗：去零宽/不间断空格、合并空行、行首空白规整。"""
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [line.strip() for line in text.splitlines()]
    out: list[str] = []
    prev_blank = True  # 开头不保留空行
    for line in lines:
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def render_markdown(law: LawSource, body: str) -> str:
    """front matter + 正文。front matter 即入库 metadata 的唯一来源。"""
    front = {
        "title": law.title,
        "level": law.level,
        "issuing_authority": law.issuing_authority,
        "validity": law.validity,
        "effective_date": law.effective_date,
        "source_url": law.url,
    }
    return "---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + "---\n\n" + body


def fetch(url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        return resp.text


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for law in LAWS:
        body = extract_body(fetch(law.url), law.container_class)
        article_count = len(re.findall(r"^第[一二三四五六七八九十百千零]+条[　\s]", body, re.M))
        if article_count == 0:
            raise ValueError(f"{law.law_id}: 未解析到任何条文，页面结构可能已变化")
        content = render_markdown(law, body)
        target = DATA_DIR / f"{law.law_id}.md"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=DATA_DIR, delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(content)
        Path(tmp.name).replace(target)
        print(f"[OK] {law.law_id}: {law.title} | {article_count} 条 | {len(body)} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
