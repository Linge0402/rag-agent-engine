"""技术文档领域包：开发者官方文档知识库（首个实例：LangGraph）。

W5 验收标准：接入本包**不改 core 一行代码**——解析器、切分装饰器、
工具、模板、评估集全部在本目录内通过 KnowledgePack 协议声明。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.agent.tools import Tool
from core.chunking.base import ChunkStrategy
from core.parsing.base import Parser
from core.service.schemas import EvalCase, EvalSet, FilterField, FilterSchema, PromptTemplate
from packs.devdocs.parsers import DevDocsChunkStrategy, DevDocsMarkdownParser
from packs.devdocs.prompts import DEVDOCS_TEMPLATE
from packs.devdocs.tools import RunPythonTool, VersionCompatTool

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "devdocs"
EVAL_PATH = Path(__file__).resolve().parent / "eval" / "devdocs_eval.yaml"

# 检索默认硬过滤：只检索当前库版本的文档。
# 多版本文档共存时（未来扩展），版本过滤防止旧版 API 答案污染。
DEFAULT_RETRIEVAL_FILTERS: dict[str, str] = {}

# 相关性阈值（bge-m3 + 评估集校准，2026-10-11，按检索层实际生效的原始向量分）：
# 相关问题 top1 原始分 0.586~0.71，边界问题 0.43~0.54（PyTorch 类问题与
# 流式/加载概念语义相近，分数偏高）；阈值取 0.56。
# 注意：校准必须用 store.search 的原始向量分，rerank 调整后的分数口径不同。
# 换 embedding 模型或扩评估集后重新校准。
MIN_RETRIEVAL_SCORE = 0.56

# 技术文档问答无需免责声明（与脚本取值口径保持一致，显式置空）
DISCLAIMER = ""


class DevDocsPack:
    """技术文档领域包。"""

    name = "devdocs"

    def __init__(self, data_dir: Path = DATA_DIR, store=None) -> None:
        self._data_dir = data_dir
        self._store = store

    def parsers(self) -> list[Parser]:
        return [DevDocsMarkdownParser()]

    def chunk_strategy(self) -> ChunkStrategy:
        return DevDocsChunkStrategy()

    def filter_schema(self) -> FilterSchema:
        return FilterSchema(
            fields=[
                FilterField(name="version", type="keyword", description="库版本号"),
                FilterField(
                    name="extra.library", type="keyword", description="库名，如 langgraph"
                ),
                FilterField(
                    name="extra.doc_url", type="keyword", description="文档页 URL"
                ),
            ]
        )

    def tools(self) -> list[Tool]:
        return [VersionCompatTool(), RunPythonTool()]

    def prompt_template(self) -> PromptTemplate:
        return DEVDOCS_TEMPLATE

    def eval_sets(self) -> list[EvalSet]:
        data = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
        return [
            EvalSet(
                name=data["name"],
                cases=[EvalCase.model_validate(c) for c in data["cases"]],
            )
        ]
