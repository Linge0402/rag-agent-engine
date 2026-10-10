"""法律领域 Agent 工具：时效校验、条文定位、新旧法对比。

三个工具对应法律问答的三类刚性需求：
- check_validity：用户引用的法规是否现行有效——过期法规的建议是有害的；
- locate_article：精确定位"某法第 N 条"——确定性 chunk_id 让定位不需要
  语义检索，直接按 ID 取原文（快且不会错）；
- compare_versions：新旧法差异——以公司法 2018/2023 两版为典型场景。

工具协议见 core/agent/tools.py：run() 不抛异常，失败走 ToolResult.error。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.agent.tools import QueryContext, ToolResult
from packs.legal.parsers import LegalChunkStrategy, LegalMarkdownParser
from packs.legal.registry import LawRecord, LawRegistry

# ---------------------------------------------------------------------------
# 时效校验
# ---------------------------------------------------------------------------


class CheckValidityArgs(BaseModel):
    law_name: str = Field(description="法规名称，如「公司法」「中华人民共和国民法典」")


class CheckValidityTool:
    """校验一部法规的时效状态与施行日期。"""

    name = "check_validity"
    description = (
        "查询一部法律法规的时效状态（现行有效/已修订/已废止）、效力级别、"
        "发文机关与施行日期。当用户提到某部法规、或准备引用某部法规作答前，"
        "先用本工具确认其仍然现行有效。"
    )
    args_schema = CheckValidityArgs

    def __init__(self, registry: LawRegistry) -> None:
        self._registry = registry

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, CheckValidityArgs)
        records = self._registry.find(args.law_name)
        if not records:
            return ToolResult(error=f"知识库中未收录「{args.law_name}」")
        return ToolResult(
            data={
                "results": [_record_dict(r) for r in records],
                "hint": "存在多个版本时，仅「现行有效」版本可用于回答。",
            }
        )


# ---------------------------------------------------------------------------
# 条文定位
# ---------------------------------------------------------------------------


class LocateArticleArgs(BaseModel):
    law_name: str = Field(description="法规名称，如「民法典」")
    article_no: int = Field(description="条号（阿拉伯数字），如 188 表示第一百八十八条", gt=0)


class LocateArticleTool:
    """按"法规 + 条号"精确定位条文原文。

    走确定性 chunk_id（{law_id}-t{条号:04d}）直接取原文，
    不经语义检索——条文定位是精确查找，向量检索只会引入不确定性。
    """

    name = "locate_article"
    description = (
        "定位某部法规的第 N 条原文。当用户问「XX法第 N 条是什么」、"
        "或需要逐字核对某条条文时使用。"
    )
    args_schema = LocateArticleArgs

    def __init__(self, registry: LawRegistry, store) -> None:
        self._registry = registry
        self._store = store  # QdrantVectorStore；测试可注入带 get() 的 Fake

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, LocateArticleArgs)
        records = self._registry.find(args.law_name)
        if not records:
            return ToolResult(error=f"知识库中未收录「{args.law_name}」")
        current = [r for r in records if r.validity == "现行有效"] or records
        record = current[0]
        if len(records) > 1:
            others = "、".join(f"{r.title}（{r.validity}）" for r in records if r is not record)
            note = f"该法规存在其他版本：{others}；以下条文取自 {record.title}。"
        else:
            note = ""
        chunk = self._store.get(f"{record.law_id}-t{args.article_no:04d}")
        if chunk is None:
            return ToolResult(
                error=f"{record.title} 中不存在第 {args.article_no} 条（或该法规尚未入库）"
            )
        return ToolResult(chunks=[chunk], data={"note": note, "validity": record.validity})


# ---------------------------------------------------------------------------
# 新旧法对比
# ---------------------------------------------------------------------------


class CompareVersionsArgs(BaseModel):
    law_name: str = Field(description="法规名称（不带版本年份），如「公司法」")
    article_no: int | None = Field(
        default=None, description="可选：聚焦对比某一条号（按新法条号）", gt=0
    )


class CompareVersionsTool:
    """对比同一部法规的不同版本。

    数据源是法规文件本身（registry 记录的 path），而非向量库——
    对比需要完整条文序列，重解析文件比从向量库回捞更可靠。
    注意：修订后条文可能重新编号（如公司法 2018→2023），
    条号相同的条文仅表示位置对应，工具会如实标注这一限制。
    """

    name = "compare_versions"
    description = (
        "对比同一部法规新旧版本的差异：各版本时效状态、条文总数、"
        "以及按条号的文本异同。用户问「新公司法改了什么」类问题时使用。"
    )
    args_schema = CompareVersionsArgs

    def __init__(self, registry: LawRegistry) -> None:
        self._registry = registry

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, CompareVersionsArgs)
        records = self._registry.find(args.law_name)
        if len(records) < 2:
            return ToolResult(error=f"「{args.law_name}」在知识库中不足两个版本，无法对比")

        parser, strategy = LegalMarkdownParser(), LegalChunkStrategy()
        versions: dict[str, dict[int, str]] = {}
        for r in records:
            chunks = strategy.chunk(parser.parse(r.path))
            # chunk 首行是面包屑（含法规标题，各版本必然不同），对比应只比正文
            versions[r.law_id] = {
                c.metadata.extra["article_no"]: c.text.split("\n", 1)[1] for c in chunks
            }

        ids = list(versions)

        def effective_date(law_id: str) -> str:
            return next(r for r in records if r.law_id == law_id).effective_date

        old_id, new_id = sorted(ids, key=effective_date)
        old, new = versions[old_id], versions[new_id]

        changed = sorted(no for no in old.keys() & new.keys() if old[no] != new[no])
        only_old = sorted(old.keys() - new.keys())
        only_new = sorted(new.keys() - old.keys())

        data: dict = {
            "old": _record_dict(next(r for r in records if r.law_id == old_id)),
            "new": _record_dict(next(r for r in records if r.law_id == new_id)),
            "old_article_count": len(old),
            "new_article_count": len(new),
            "same_no_changed_count": len(changed),
            "only_in_old": only_old[:50],
            "only_in_new": only_new[:50],
            "caveat": "修订可能伴随条文重新编号，条号相同不必然表示内容对应关系。",
        }
        if args.article_no is not None:
            no = args.article_no
            data["focus"] = {
                "article_no": no,
                "old_text": old.get(no),
                "new_text": new.get(no),
                "changed": old.get(no) is not None and old.get(no) != new.get(no),
            }
        return ToolResult(data=data)


def _record_dict(r: LawRecord) -> dict:
    return {
        "law_id": r.law_id,
        "title": r.title,
        "level": r.level,
        "issuing_authority": r.issuing_authority,
        "validity": r.validity,
        "effective_date": r.effective_date,
    }
