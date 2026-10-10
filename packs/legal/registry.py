"""法规登记簿：data/legal 目录下全部法规的 metadata 索引。

时效校验、新旧法对比等工具的数据源。登记簿从法规文件的 front matter
构建（与入库 metadata 同源），避免维护两份会漂移的数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packs.legal.parsers import read_front_matter


@dataclass(frozen=True)
class LawRecord:
    """一部法规（一个版本）的登记信息。"""

    law_id: str  # 文件 stem，如 minfadian / gongsifa-2018
    title: str
    level: str
    issuing_authority: str
    validity: str
    effective_date: str
    path: Path


class LawRegistry:
    """法规登记簿。构建时只读 front matter，不解析正文（保持轻量）。"""

    def __init__(self, data_dir: Path) -> None:
        self._records: list[LawRecord] = []
        for path in sorted(data_dir.glob("*.md")):
            meta = read_front_matter(path.read_text(encoding="utf-8"))
            if meta is None:
                continue
            self._records.append(
                LawRecord(
                    law_id=path.stem,
                    title=str(meta.get("title", path.stem)),
                    level=str(meta.get("level", "")),
                    issuing_authority=str(meta.get("issuing_authority", "")),
                    validity=str(meta.get("validity", "")),
                    effective_date=str(meta.get("effective_date", "")),
                    path=path,
                )
            )

    def all(self) -> list[LawRecord]:
        return list(self._records)

    def find(self, name: str) -> list[LawRecord]:
        """按名称查找：先精确匹配标题，再退化到包含式模糊匹配（"公司法"命中两版）。"""
        name = name.strip().strip("《》")
        exact = [r for r in self._records if r.title == name or r.law_id == name]
        if exact:
            return exact
        return [r for r in self._records if name in r.title]
