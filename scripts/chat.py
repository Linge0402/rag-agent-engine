"""终端法律问答机器人（法律领域包）。

用法：
    uv run python scripts/chat.py [法规目录]   # 默认 data/legal

- 首次运行会先入库（幂等，重复运行不会重复入库）；
- 检索默认硬过滤"现行有效"——过期法规留在库里供对比工具使用，
  但不进入检索视野（见 packs/legal/__init__.py）；
- 多轮对话：最近 5 轮窗口 + 长期事实记忆；
- 输入 exit / quit / 退出 结束。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.agent.graph import AgentDeps, build_graph
from core.agent.memory import Memory
from core.agent.tools import QueryContext
from core.config import get_settings
from core.generation.llm import OpenAIChatClient
from core.indexing.embeddings import OpenAIEmbeddingClient
from core.indexing.ingestion import ingest_paths
from core.indexing.vector_store import QdrantVectorStore
from core.retrieval.filters import FilterCondition, FilterSet
from packs.legal import DATA_DIR, DEFAULT_RETRIEVAL_FILTERS, MIN_RETRIEVAL_SCORE, LegalPack

CHAT_COLLECTION = "legal"
HISTORY_TURNS = 5  # 多轮记忆窗口：保留最近 5 轮


def build_search_filters(pack: LegalPack) -> dict:
    """把 pack 的默认检索过滤过一遍 DSL 校验——校验不过宁可报错也不静默放行。"""
    fs = FilterSet(
        conditions=[
            FilterCondition(field=k, op="eq", value=v) for k, v in DEFAULT_RETRIEVAL_FILTERS.items()
        ]
    )
    problems = fs.validate(pack.filter_schema())
    if problems:
        raise ValueError(f"检索过滤条件未通过 pack schema 校验: {problems}")
    return fs.to_store_conditions()


def main() -> int:
    doc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR

    settings = get_settings()
    embedder = OpenAIEmbeddingClient.from_settings(settings)
    llm = OpenAIChatClient.from_settings(settings)
    store = QdrantVectorStore(settings.qdrant_url, collection=CHAT_COLLECTION, dim=embedder.dim)
    store.ensure_collection()

    pack = LegalPack(store=store)

    paths = sorted(doc_dir.glob("**/*.md"))
    if paths:
        report = ingest_paths(paths, pack.parsers(), pack.chunk_strategy(), embedder, store)
        print(f"[入库] {report.docs} 个文档 → {report.chunks} 个 chunk\n")

    graph = build_graph(
        AgentDeps(
            embedder=embedder,
            store=store,
            llm=llm,
            template=pack.prompt_template(),
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
            search_filters=build_search_filters(pack),
            min_score=MIN_RETRIEVAL_SCORE,
        )
    )

    tools = {t.name: t for t in pack.tools()}
    ctx = QueryContext(
        session_id="chat", pack_name=pack.name, pack_filter_schema=pack.filter_schema()
    )

    print("法律知识助手（仅供参考，不构成法律意见）。")
    print("斜杠命令：/时效 <法规名>  /条文 <法规名> <条号>  /对比 <法规名>；输入 exit 退出。\n")
    history: list[dict[str, str]] = []
    memory = Memory()  # 长期记忆：跨会话窗口记住用户事实
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

        if question.startswith("/"):
            _run_tool_command(question, tools, ctx)
            continue

        state = graph.invoke({"question": question, "history": history, "facts": memory.facts})
        memory.facts = state.get("facts", memory.facts)

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


def _run_tool_command(command: str, tools: dict, ctx: QueryContext) -> None:
    """斜杠命令直连领域工具（绕过闲聊路由，演示工具的确性输出）。"""
    parts = command.split()
    name_map = {"/时效": "check_validity", "/条文": "locate_article", "/对比": "compare_versions"}
    tool_name = name_map.get(parts[0])
    if tool_name is None or tool_name not in tools:
        print(f"未知命令。可用：{'、'.join(name_map)}")
        return
    tool = tools[tool_name]
    try:
        if tool_name == "check_validity":
            args = tool.args_schema(law_name=" ".join(parts[1:]))
        elif tool_name == "locate_article":
            args = tool.args_schema(law_name=parts[1], article_no=int(parts[2]))
        else:
            args = tool.args_schema(law_name=" ".join(parts[1:]))
    except (IndexError, ValueError) as e:
        print(f"参数不对: {e}")
        return
    result = asyncio.run(tool.run(args, ctx))
    if not result.ok:
        print(f"工具返回错误：{result.error}\n")
        return
    for chunk in result.chunks:
        print(f"{chunk.text}\n")
    if result.data:
        import json

        print(json.dumps(result.data, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    sys.exit(main())
