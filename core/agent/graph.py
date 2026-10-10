"""LangGraph 状态图：意图路由 → 闲聊 / 法律问答 / 敏感拒答。

设计要点：
- 路由节点用 LLM 做意图分类（同一模型，不引入额外依赖）；
- 闲聊与法律问答分流：闲聊不检索知识库，法律问题走完整 RAG 链路；
- 法律回答固定附免责声明——这是领域边界设计，不是示弱；
- 图本身无状态：多轮历史由调用方（scripts/chat.py）持有并传入，
  历史压缩与查询改写在后续节点中扩展（见 memory.py / 改写节点规划）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from core.generation.llm import LLMClient, build_answer
from core.generation.prompt import DEFAULT_TEMPLATE, build_messages
from core.indexing.embeddings import EmbeddingClient
from core.indexing.vector_store import QdrantVectorStore, ScoredChunk
from core.retrieval.rerank import rerank
from core.service.schemas import Answer, PromptTemplate

Intent = Literal["chitchat", "legal", "sensitive"]

DISCLAIMER = "\n\n（以上回答仅供参考，不构成法律意见；具体问题请咨询执业律师。）"

SENSITIVE_REFUSAL = (
    "抱歉，这个请求我无法协助。我不能帮助规避法律或可能违法的行为。"
    "如果你面临具体的法律纠纷，建议咨询执业律师获得专业帮助。"
)

_INTENT_PROMPT = """判断用户输入属于哪一类，只回复类别词本身，不要任何解释：
- chitchat：问候、寒暄、闲聊，或询问"你是谁"之类的日常对话
- legal：与法律、法规、权利义务相关的问题
- sensitive：请求协助违法、规避法律或伤害他人的行为

用户输入：{question}"""

_CHITCHAT_SYSTEM = (
    "你是一个法律知识助手的闲聊人格。友好、简洁地回应用户的寒暄，"
    "并在自然的时候说明：你专门回答中国法律法规相关问题，回答仅供参考、不构成法律意见。"
    "不要主动展开法律分析；用户没问法律问题时不要硬答。"
)


class AgentState(TypedDict, total=False):
    """一次提问在状态图中流转的全部数据。"""

    question: str
    history: list[dict[str, str]]  # [{"role": "user"/"assistant", "content": ...}]
    intent: Intent
    retrieved: list[ScoredChunk]
    answer: Answer | None  # 法律路径的产物
    reply: str | None  # 闲聊/拒答路径的产物


def parse_intent(text: str) -> Intent:
    """解析 LLM 的分类输出。解析失败默认走 legal——宁可检索后拒答，也不瞎聊。"""
    cleaned = text.strip().lower()
    for intent in ("chitchat", "sensitive", "legal"):
        if intent in cleaned:
            return intent  # type: ignore[return-value]
    return "legal"


@dataclass
class AgentDeps:
    """状态图节点的全部外部依赖，构造时注入（测试注入 Fake）。"""

    embedder: EmbeddingClient
    store: QdrantVectorStore
    llm: LLMClient
    template: PromptTemplate = field(default_factory=lambda: DEFAULT_TEMPLATE)
    retrieve_top_k: int = 20
    rerank_top_k: int = 6


def build_graph(deps: AgentDeps):
    """组装并编译状态图。"""

    def route(state: AgentState) -> dict:
        messages = [{"role": "user", "content": _INTENT_PROMPT.format(question=state["question"])}]
        return {"intent": parse_intent(deps.llm.chat(messages))}

    def chitchat(state: AgentState) -> dict:
        messages = [{"role": "system", "content": _CHITCHAT_SYSTEM}]
        messages.extend(state.get("history", []))
        messages.append({"role": "user", "content": state["question"]})
        return {"reply": deps.llm.chat(messages)}

    def sensitive(state: AgentState) -> dict:
        return {"reply": SENSITIVE_REFUSAL}

    def retrieve(state: AgentState) -> dict:
        vector = deps.embedder.embed([state["question"]])[0]
        candidates = deps.store.search(vector, top_k=deps.retrieve_top_k)
        top = rerank(state["question"], candidates, top_k=deps.rerank_top_k)
        return {"retrieved": top}

    def generate(state: AgentState) -> dict:
        import time

        chunks = [hit.chunk for hit in state["retrieved"]]
        messages = build_messages(deps.template, chunks, state["question"])
        start = time.perf_counter()
        text = deps.llm.chat(messages)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = build_answer(text, chunks, latency_ms)
        if answer.status == "answered":
            answer.text += DISCLAIMER
        return {"answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("chitchat", chitchat)
    graph.add_node("sensitive", sensitive)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["intent"],
        {"chitchat": "chitchat", "legal": "retrieve", "sensitive": "sensitive"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("chitchat", END)
    graph.add_edge("sensitive", END)
    graph.add_edge("generate", END)
    return graph.compile()
