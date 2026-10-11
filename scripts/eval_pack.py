"""领域包评估基线：对评估集逐条跑检索 + 生成，输出指标报告。

用法：
    uv run python scripts/eval_pack.py [--pack legal|devdocs] [--no-llm] [--out report.json]

--no-llm：只评估检索（命中/MRR），不调 LLM——检索是生成的上游，
上游指标不达标时先修检索，不看生成指标（避免归因混乱）。

报告同时打印到终端并（可选）写 JSON，供 README 引用与历次对比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.agent.graph import AgentDeps, build_graph
from core.config import get_settings
from core.evaluation.answer_metrics import citation_accuracy, is_correct_refusal
from core.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank
from core.generation.llm import OpenAIChatClient
from core.indexing.embeddings import OpenAIEmbeddingClient
from core.indexing.vector_store import QdrantVectorStore
from packs import devdocs, legal

PACKS = {
    "legal": {
        "factory": legal.LegalPack,
        "default_filters": legal.DEFAULT_RETRIEVAL_FILTERS,
        "min_score": legal.MIN_RETRIEVAL_SCORE,
        "disclaimer": legal.DISCLAIMER,
    },
    "devdocs": {
        "factory": devdocs.DevDocsPack,
        "default_filters": devdocs.DEFAULT_RETRIEVAL_FILTERS,
        "min_score": devdocs.MIN_RETRIEVAL_SCORE,
        "disclaimer": devdocs.DISCLAIMER,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", choices=sorted(PACKS), default="legal", help="领域包")
    ap.add_argument("--no-llm", action="store_true", help="只评估检索指标")
    ap.add_argument("--out", type=Path, default=None, help="报告 JSON 输出路径")
    args = ap.parse_args()

    spec = PACKS[args.pack]
    settings = get_settings()
    pack = spec["factory"]()
    embedder = OpenAIEmbeddingClient.from_settings(settings)
    llm = OpenAIChatClient.from_settings(settings)
    store = QdrantVectorStore(settings.qdrant_url, collection=args.pack, dim=embedder.dim)

    graph = build_graph(
        AgentDeps(
            embedder=embedder,
            store=store,
            llm=llm,
            template=pack.prompt_template(),
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
            search_filters=dict(spec["default_filters"]),
            min_score=spec["min_score"],
            disclaimer=spec["disclaimer"],
        )
    )

    eval_set = pack.eval_sets()[0]
    rows: list[dict] = []
    for case in eval_set.cases:
        state = graph.invoke({"question": case.question, "history": []})
        retrieved = state.get("retrieved", [])
        retrieved_ids = [hit.chunk.chunk_id for hit in retrieved]
        answer = state.get("answer")

        row: dict = {"question": case.question}
        if case.expected_chunk_ids:
            row["hit@k"] = hit_at_k(retrieved_ids, case.expected_chunk_ids, len(retrieved_ids) or 1)
            row["rr"] = round(reciprocal_rank(retrieved_ids, case.expected_chunk_ids), 3)
        if not args.no_llm and answer is not None:
            chunks = [hit.chunk for hit in retrieved]
            row["refusal_ok"] = is_correct_refusal(answer, case.should_refuse)
            if not case.should_refuse:
                row["citation_acc"] = citation_accuracy(answer, chunks)
        rows.append(row)
        print(f"  {case.question[:30]:<32} {row}")

    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    report = {
        "pack": args.pack,
        "eval_set": eval_set.name,
        "cases": len(rows),
        "metrics": {
            "hit_rate": avg("hit@k"),
            "mrr": avg("rr"),
            "citation_accuracy": avg("citation_acc"),
            "refusal_accuracy": avg("refusal_ok"),
        },
        "rows": rows,
    }
    print("\n=== 基线报告 ===")
    for k, v in report["metrics"].items():
        if v is not None:
            print(f"  {k}: {v}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
