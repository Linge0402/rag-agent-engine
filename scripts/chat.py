"""终端知识库问答机器人：一个引擎，多个领域包。

用法：
    uv run python scripts/chat.py [--pack legal|devdocs] [文档目录]

- 每个领域包一个独立 collection（按包名命名），首次运行自动入库（幂等）；
- 检索过滤与相关性阈值由各 pack 声明（如法律包"现行有效"硬过滤）；
- 注入工具表后，Agent 自行决定"检索知识库"还是"调用领域工具"；
- 多轮对话：最近 5 轮窗口 + 长期事实记忆；
- 输入 exit / quit / 退出 结束。
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from packs import devdocs, legal

HISTORY_TURNS = 5  # 多轮记忆窗口：保留最近 5 轮

# 领域包注册表：接入新领域只需在这里加一行
PACKS = {
    "legal": {
        "factory": legal.LegalPack,
        "data_dir": legal.DATA_DIR,
        "default_filters": legal.DEFAULT_RETRIEVAL_FILTERS,
        "min_score": legal.MIN_RETRIEVAL_SCORE,
        "disclaimer": legal.DISCLAIMER,
        "greeting": "法律知识助手（仅供参考，不构成法律意见）。",
        "commands": {
            "/时效": "check_validity",
            "/条文": "locate_article",
            "/对比": "compare_versions",
        },
    },
    "devdocs": {
        "factory": devdocs.DevDocsPack,
        "data_dir": devdocs.DATA_DIR,
        "default_filters": devdocs.DEFAULT_RETRIEVAL_FILTERS,
        "min_score": devdocs.MIN_RETRIEVAL_SCORE,
        "disclaimer": devdocs.DISCLAIMER,
        "greeting": "LangGraph 文档助手（基于官方文档，版本对齐本地安装）。",
        "commands": {"/兼容": "check_version_compat"},
    },
}


def build_search_filters(pack, default_filters: dict) -> dict:
    """把 pack 的默认检索过滤过一遍 DSL 校验——校验不过宁可报错也不静默放行。"""
    fs = FilterSet(
        conditions=[FilterCondition(field=k, op="eq", value=v) for k, v in default_filters.items()]
    )
    problems = fs.validate(pack.filter_schema())
    if problems:
        raise ValueError(f"检索过滤条件未通过 pack schema 校验: {problems}")
    return fs.to_store_conditions()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", choices=sorted(PACKS), default="legal", help="领域包")
    ap.add_argument(
        "doc_dir", nargs="?", type=Path, default=None, help="文档目录（默认取包内数据目录）"
    )
    args = ap.parse_args()

    spec = PACKS[args.pack]
    doc_dir = args.doc_dir or spec["data_dir"]

    settings = get_settings()
    embedder = OpenAIEmbeddingClient.from_settings(settings)
    llm = OpenAIChatClient.from_settings(settings)
    store = QdrantVectorStore(settings.qdrant_url, collection=args.pack, dim=embedder.dim)
    store.ensure_collection()

    pack = spec["factory"](store=store)

    paths = sorted(doc_dir.glob("**/*.md"))
    if paths:
        report = ingest_paths(paths, pack.parsers(), pack.chunk_strategy(), embedder, store)
        print(f"[入库] {report.docs} 个文档 → {report.chunks} 个 chunk\n")

    tools = {t.name: t for t in pack.tools()}
    ctx = QueryContext(
        session_id="chat", pack_name=pack.name, pack_filter_schema=pack.filter_schema()
    )

    graph = build_graph(
        AgentDeps(
            embedder=embedder,
            store=store,
            llm=llm,
            template=pack.prompt_template(),
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
            search_filters=build_search_filters(pack, spec["default_filters"]),
            min_score=spec["min_score"],
            disclaimer=spec["disclaimer"],
            tools=tools,  # 注入工具表后，Agent 自行决定检索还是调工具
            tool_ctx=ctx,
        )
    )

    print(spec["greeting"])
    if spec["commands"]:
        print(f"斜杠命令：{'  '.join(spec['commands'])}（也可直接提问，Agent 会自行决定）")
    print("输入 exit 退出。\n")
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
            _run_tool_command(question, tools, ctx, spec["commands"])
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


def _run_tool_command(command: str, tools: dict, ctx: QueryContext, commands: dict) -> None:
    """斜杠命令直连领域工具（绕过 Agent 决策，演示工具的确定性输出）。"""
    parts = command.split()
    tool_name = commands.get(parts[0])
    if tool_name is None or tool_name not in tools:
        print(f"未知命令。可用：{'、'.join(commands)}")
        return
    tool = tools[tool_name]
    try:
        if tool_name == "check_validity":
            args = tool.args_schema(law_name=" ".join(parts[1:]))
        elif tool_name == "locate_article":
            args = tool.args_schema(law_name=parts[1], article_no=int(parts[2]))
        elif tool_name == "compare_versions":
            args = tool.args_schema(law_name=" ".join(parts[1:]))
        elif tool_name == "check_version_compat":
            args = tool.args_schema(library=parts[1], version=parts[2] if len(parts) > 2 else "")
        else:
            print(f"命令 {parts[0]} 暂未接入")
            return
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
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    sys.exit(main())
