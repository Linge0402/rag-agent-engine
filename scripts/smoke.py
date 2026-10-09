"""端到端冒烟：入库一批 Markdown → 提问 → 输出带引用的回答。

用法：
    uv run python scripts/check_env.py            # 先确认端点就绪
    uv run python scripts/smoke.py <文档目录> "你的问题"

示例：
    uv run python scripts/smoke.py data/langgraph-docs "StateGraph 怎么定义节点？"

说明：
- 端到端需要真实端点（Qdrant / embedding / LLM），配置见 .env；
- 冒烟数据固定写入 "smoke" collection，不影响其他数据；
- 重复执行是幂等的（point id 由 chunk_id 确定性生成）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from core.chunking.structural import StructuralChunkStrategy
from core.config import get_settings
from core.generation.llm import OpenAIChatClient, build_answer, validate_citations
from core.generation.prompt import DEFAULT_TEMPLATE, build_messages
from core.indexing.embeddings import OpenAIEmbeddingClient
from core.indexing.ingestion import ingest_paths
from core.indexing.vector_store import QdrantVectorStore
from core.parsing.loaders import DEFAULT_PARSERS
from core.retrieval.rerank import rerank

SMOKE_COLLECTION = "smoke"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    doc_dir, question = Path(sys.argv[1]), sys.argv[2]
    if not doc_dir.is_dir():
        print(f"文档目录不存在: {doc_dir}")
        return 2

    settings = get_settings()
    embedder = OpenAIEmbeddingClient.from_settings(settings)
    llm = OpenAIChatClient.from_settings(settings)
    store = QdrantVectorStore(settings.qdrant_url, collection=SMOKE_COLLECTION, dim=embedder.dim)
    store.ensure_collection()

    # ---- 入库 ----
    paths = sorted(doc_dir.glob("**/*.md"))
    if not paths:
        print(f"{doc_dir} 下没有 Markdown 文件")
        return 2
    report = ingest_paths(paths, DEFAULT_PARSERS, StructuralChunkStrategy(), embedder, store)
    print(f"[入库] {report.docs} 个文档 → {report.chunks} 个 chunk")
    for error in report.errors:
        print(f"  跳过: {error}")

    # ---- 检索 ----
    query_vector = embedder.embed([question])[0]
    candidates = store.search(query_vector, top_k=settings.retrieve_top_k)
    top = rerank(question, candidates, top_k=settings.rerank_top_k)
    print(f"[检索] 召回 {len(candidates)} 条，重排后取 {len(top)} 条")
    for hit in top:
        print(f"  {hit.score:.3f}  {hit.chunk.chunk_id}  {hit.chunk.text[:40]}…")

    # ---- 生成 ----
    chunks = [hit.chunk for hit in top]
    messages = build_messages(DEFAULT_TEMPLATE, chunks, question)
    start = time.perf_counter()
    text = llm.chat(messages)
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = build_answer(text, chunks, latency_ms)

    # ---- 输出 ----
    print(f"\n[回答]（status={answer.status}, confidence={answer.confidence:.2f}, "
          f"latency={answer.latency_ms}ms）")
    print(answer.text)
    if answer.citations:
        print("\n[引用]")
        for citation in answer.citations:
            print(f"  {citation.chunk_id}  来源：{citation.source}")
        problems = validate_citations(answer, chunks)
        print("\n[引用校验]", "全部通过" if not problems else f"{len(problems)} 个问题: {problems}")

    return 0 if answer.status == "answered" else 1


if __name__ == "__main__":
    sys.exit(main())
