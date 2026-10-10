"""终端法律问答机器人。

用法：
    uv run python scripts/chat.py [文档目录]

- 首次使用可传入文档目录先入库（幂等，重复运行不会重复入库）；
- 之后直接对话；输入 exit / quit / 退出 结束；
- 多轮历史保留最近 5 轮。
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.agent.graph import AgentDeps, build_graph
from core.chunking.structural import StructuralChunkStrategy
from core.config import get_settings
from core.generation.llm import OpenAIChatClient
from core.indexing.embeddings import OpenAIEmbeddingClient
from core.indexing.ingestion import ingest_paths
from core.indexing.vector_store import QdrantVectorStore
from core.parsing.loaders import DEFAULT_PARSERS

CHAT_COLLECTION = "smoke"  # 与 smoke.py 共用一个 collection
HISTORY_TURNS = 5  # 多轮记忆窗口：保留最近 5 轮


def main() -> int:
    doc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    settings = get_settings()
    embedder = OpenAIEmbeddingClient.from_settings(settings)
    llm = OpenAIChatClient.from_settings(settings)
    store = QdrantVectorStore(settings.qdrant_url, collection=CHAT_COLLECTION, dim=embedder.dim)
    store.ensure_collection()

    if doc_dir is not None:
        paths = sorted(doc_dir.glob("**/*.md"))
        report = ingest_paths(paths, DEFAULT_PARSERS, StructuralChunkStrategy(), embedder, store)
        print(f"[入库] {report.docs} 个文档 → {report.chunks} 个 chunk\n")

    graph = build_graph(
        AgentDeps(
            embedder=embedder,
            store=store,
            llm=llm,
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
        )
    )

    print("法律知识助手（仅供参考，不构成法律意见）。输入 exit 退出。\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question in {"exit", "quit", "退出"}:
            print("再见！")
            break

        state = graph.invoke({"question": question, "history": history})

        if state.get("answer") is not None:
            answer = state["answer"]
            print(f"助手: {answer.text}")
            if answer.citations:
                sources = "、".join(f"{c.chunk_id}" for c in answer.citations)
                print(f"  （引用：{sources} | 状态：{answer.status}）")
            reply_text = answer.text
        else:
            reply_text = state["reply"] or "……"
            print(f"助手: {reply_text}")
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply_text})
        history = history[-HISTORY_TURNS * 2 :]

    return 0


if __name__ == "__main__":
    sys.exit(main())
