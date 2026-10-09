"""切分策略抽象。

引擎内置 `structural.py` 作为默认实现（标题/段落结构切分）；
领域包可以提供自己的实现（如法规的"编-章-节-条"切分），
通过 KnowledgePack.chunk_strategy() 注入入库流水线。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.service.schemas import Chunk, ParsedDocument


class ChunkStrategy(ABC):
    """把解析后的文档切成带 metadata 的 Chunk 序列。

    实现要求：
    - 确定性：同一输入必须产生相同的 chunk_id 序列（幂等写入依赖这一点）；
    - metadata 完整性：每个 Chunk 至少填好 source 与 section_path；
    - 不丢内容：所有 Chunk 的 text 拼接应能无损覆盖原文（允许的例外：
      清洗阶段已移除的页眉页脚等噪声）。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名，用于日志与评估报告。"""

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """切分单个文档。"""
