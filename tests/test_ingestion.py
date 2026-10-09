"""入库编排测试：Fake embedder + Fake store，离线可跑。"""

from pathlib import Path

from core.chunking.structural import StructuralChunkStrategy
from core.indexing.ingestion import ingest_paths
from core.parsing.loaders import DEFAULT_PARSERS


class FakeEmbedder:
    """固定 4 维假向量，离线测试用。"""

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    """内存版 store，只记录写入了什么。"""

    def __init__(self):
        self.written: list[str] = []

    def upsert(self, chunks, vectors) -> int:
        assert len(chunks) == len(vectors)
        self.written.extend(c.chunk_id for c in chunks)
        return len(chunks)


def test_ingest_markdown_files(tmp_path: Path):
    (tmp_path / "民法典.md").write_text("# 总则\n\n第一条 为了保护民事主体的合法权益。\n")
    (tmp_path / "刑法.md").write_text("# 总则\n\n第一条 为了惩罚犯罪，保护人民。\n")

    store = FakeStore()
    report = ingest_paths(
        sorted(tmp_path.glob("*.md")),
        DEFAULT_PARSERS,
        StructuralChunkStrategy(),
        FakeEmbedder(),
        store,
    )

    assert report.ok
    assert report.docs == 2
    assert report.chunks == len(store.written)
    assert "民法典-c001" in store.written


def test_ingest_skips_unsupported_files(tmp_path: Path):
    (tmp_path / "民法典.md").write_text("# 总则\n\n第一条 内容。\n")
    (tmp_path / "汇编.pdf").write_bytes(b"%PDF-fake")

    report = ingest_paths(
        sorted(tmp_path.iterdir()),
        DEFAULT_PARSERS,
        StructuralChunkStrategy(),
        FakeEmbedder(),
        FakeStore(),
    )

    assert not report.ok
    assert report.docs == 1  # 失败文件不中断其余入库
    assert any("没有解析器支持" in e for e in report.errors)
