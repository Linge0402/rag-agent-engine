"""法律领域包：中国法律法规知识库。

实现 KnowledgePack 协议的六个声明（协议见 core/pack.py）。
引擎侧不感知任何法律细节；本包是唯一知道"编-章-节-条"的地方。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.agent.tools import Tool
from core.chunking.base import ChunkStrategy
from core.parsing.base import Parser
from core.service.schemas import EvalCase, EvalSet, FilterField, FilterSchema, PromptTemplate
from packs.legal.parsers import LegalChunkStrategy, LegalMarkdownParser
from packs.legal.prompts import LEGAL_TEMPLATE
from packs.legal.registry import LawRegistry
from packs.legal.tools import CheckValidityTool, CompareVersionsTool, LocateArticleTool

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "legal"
EVAL_PATH = Path(__file__).resolve().parent / "eval" / "legal_eval.yaml"

# 检索默认硬过滤：只检索现行有效的法规条文。
# 过期版本（如公司法 2018）留在库里供新旧法对比工具使用，但不进检索视野。
DEFAULT_RETRIEVAL_FILTERS: dict[str, str] = {"validity": "现行有效"}

# 相关性阈值（bge-m3 + 当前评估集校准，2026-10-10）：
# 相关问题 top1 分数 ≥0.75，边界问题 top1 ≤0.61；阈值取中间偏保守的 0.65。
# 换 embedding 模型或扩评估集后必须重新校准（用 scripts/eval_legal.py 验证）。
MIN_RETRIEVAL_SCORE = 0.65


class LegalPack:
    """法律法规领域包。"""

    name = "legal"

    def __init__(self, data_dir: Path = DATA_DIR, store=None) -> None:
        self._registry = LawRegistry(data_dir)
        self._store = store  # LocateArticleTool 需要；None 时该工具调用报错

    def parsers(self) -> list[Parser]:
        return [LegalMarkdownParser()]

    def chunk_strategy(self) -> ChunkStrategy:
        return LegalChunkStrategy()

    def filter_schema(self) -> FilterSchema:
        return FilterSchema(
            fields=[
                FilterField(
                    name="validity", type="keyword", description="时效状态：现行有效/已修订/已废止"
                ),
                FilterField(name="version", type="keyword", description="施行日期（版本标识）"),
                FilterField(
                    name="extra.law_id", type="keyword", description="法规 ID，如 minfadian"
                ),
                FilterField(
                    name="extra.level", type="keyword", description="效力级别：法律/行政法规等"
                ),
                FilterField(
                    name="extra.issuing_authority", type="keyword", description="发文机关"
                ),
                FilterField(
                    name="extra.article_no", type="integer", description="条号（阿拉伯数字）"
                ),
            ]
        )

    def tools(self) -> list[Tool]:
        return [
            CheckValidityTool(self._registry),
            LocateArticleTool(self._registry, self._store),
            CompareVersionsTool(self._registry),
        ]

    def prompt_template(self) -> PromptTemplate:
        return LEGAL_TEMPLATE

    def eval_sets(self) -> list[EvalSet]:
        data = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
        return [
            EvalSet(
                name=data["name"],
                cases=[EvalCase.model_validate(c) for c in data["cases"]],
            )
        ]

    @property
    def registry(self) -> LawRegistry:
        return self._registry
