"""LangGraph 状态图：意图路由 → 闲聊 / 法律问答（含工具决策）/ 敏感拒答 → 记忆更新。

设计要点：
- 路由节点用 LLM 做意图分类（同一模型，不引入额外依赖）；
- 闲聊与法律问答分流：闲聊不检索知识库，法律问题走完整 RAG 链路；
- 法律路径含改写节点：结合历史把"那它呢"改写成独立问题再检索（多轮的关键）；
- 法律路径含 plan 节点：LLM 在"检索知识库"与"调用领域工具"之间决策
  （如"公司法还有效吗"应调时效校验而非语义检索）——决策输出要求 JSON，
  解析失败一律回退到检索（与 parse_intent 同一哲学：坏输出退到安全路径）；
- 工具结果与检索结果走同一套引用校验：工具返回的 chunk 同样要求
  【chunk_id】逐字引用，严格性不因入口不同而放松；
- 双人格双温度：闲聊自然活泼（0.7），法律严谨可核验（0.0）；
- 所有路径汇入记忆节点：抽取长期事实，跨轮记住用户；
- 图本身无状态：history 与 facts 由调用方持有并传入/取回。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from core.agent.memory import Memory, extract_facts
from core.agent.tools import QueryContext, Tool, ToolResult
from core.generation.llm import LLMClient, build_answer, strip_citation_markers
from core.generation.prompt import DEFAULT_TEMPLATE, build_context, build_messages
from core.indexing.embeddings import EmbeddingClient
from core.indexing.vector_store import QdrantVectorStore, ScoredChunk
from core.retrieval.rerank import rerank
from core.service.schemas import Answer, PromptTemplate

Intent = Literal["chitchat", "legal", "sensitive"]

# 温度分流：闲聊要自然，法律要稳定
TEMPERATURE_CHITCHAT = 0.7
TEMPERATURE_LEGAL = 0.0

SENSITIVE_REFUSAL = (
    "抱歉，这个请求我无法协助。我不能帮助规避法律或可能违法的行为。"
    "如果你面临具体的法律纠纷，建议咨询执业律师获得专业帮助。"
)

_INTENT_PROMPT = """判断用户输入属于哪一类，只回复类别词本身，不要任何解释：
- chitchat：问候、寒暄、闲聊，或询问"你是谁"之类的日常对话
- legal：与法律、法规、权利义务相关的问题
- sensitive：请求协助违法、规避法律或伤害他人的行为

用户输入：{question}"""

_CHITCHAT_SYSTEM = """你是"小法"，一个法律知识助手的闲聊人格。
风格：像朋友一样自然随和，回复简短（一两句），可以有点幽默感，不说教、不机械。
边界：自然的时候说明你专门回答中国法律法规问题、回答仅供参考不构成法律意见；
不要主动展开法律分析，用户没问法律问题时不要硬答。
{memory}"""

_REWRITE_PROMPT = """你是查询改写器。根据对话历史，把用户的最新输入改写成
一个独立、完整、可直接用于检索的问题：
- 补全指代（"它""这个""那"）和省略的主语；
- 修正明显的错别字；
- 输入本身已经完整时，原样输出。
只输出改写后的问题，不要任何解释。

对话历史：
{history}

最新输入：{question}"""

_PLAN_PROMPT = """你是法律问答助手的决策器。根据用户问题，决定下一步动作：

动作一：retrieve —— 检索法规知识库回答（大多数法律问题用这个）。
动作二：tool —— 调用领域工具。可用工具：
{tool_catalog}

判断标准：只有问题明确匹配某个工具的用途时才选 tool，否则一律 retrieve。

只输出 JSON，不要任何解释：
{{"action": "retrieve"}} 或 {{"action": "tool", "tool": "<工具名>", "args": {{<参数>}}}}

用户问题：{question}"""


class AgentState(TypedDict, total=False):
    """一次提问在状态图中流转的全部数据。"""

    question: str
    history: list[dict[str, str]]  # [{"role": "user"/"assistant", "content": ...}]
    facts: list[str]  # 长期记忆：用户事实档案
    intent: Intent
    rewritten: str  # 改写后的独立问题（法律路径）
    tool_call: dict[str, Any]  # plan 节点的工具调用决策 {"tool": ..., "args": ...}
    tool_result: ToolResult  # 工具执行结果
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


def parse_plan(text: str, available_tools: set[str]) -> dict[str, Any]:
    """解析 plan 节点的 JSON 决策。

    任何解析失败（非 JSON、工具不存在、缺参数）都回退到检索路径——
    检索是安全的默认动作：错误进检索最多拒答，错误调工具可能给出
    看似权威的错误答案。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):  # 小模型常给 JSON 套代码块
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"action": "retrieve"}
    if not isinstance(plan, dict) or plan.get("action") != "tool":
        return {"action": "retrieve"}
    tool = plan.get("tool")
    args = plan.get("args")
    if tool not in available_tools or not isinstance(args, dict):
        return {"action": "retrieve"}
    return {"action": "tool", "tool": tool, "args": args}


def _render_history(history: list[dict[str, str]]) -> str:
    lines = []
    for msg in history:
        speaker = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{speaker}：{msg['content']}")
    return "\n".join(lines) or "（无历史对话）"


@dataclass
class AgentDeps:
    """状态图节点的全部外部依赖，构造时注入（测试注入 Fake）。

    search_filters：检索侧的 metadata 硬过滤（AND 等值条件），
    由调用方根据领域包的 FilterSchema 构造并经 filters.py 校验后注入；
    法律包的典型用法是 {"validity": "现行有效"}——检索层只看见现行法规。
    """

    embedder: EmbeddingClient
    store: QdrantVectorStore
    llm: LLMClient
    template: PromptTemplate = field(default_factory=lambda: DEFAULT_TEMPLATE)
    retrieve_top_k: int = 20
    rerank_top_k: int = 6
    search_filters: dict[str, Any] = field(default_factory=dict)
    # 相关性阈值：低于该向量分的召回结果丢弃；空检索 → 生成侧只能拒答。
    # 0.0 表示不过滤。取值应基于评估集校准，不可凭空设定。
    min_score: float = 0.0
    # 领域工具表：非空时法律路径启用 plan 节点（LLM 决策检索或调工具）；
    # 为空时 plan 节点直接走检索，零额外 LLM 调用。
    tools: dict[str, Tool] = field(default_factory=dict)
    tool_ctx: QueryContext | None = None
    # 回答尾部附加的领域免责声明（如法律包的"不构成法律意见"）；
    # 由调用方按领域注入——core 不感知任何领域文案。
    disclaimer: str = ""


def build_graph(deps: AgentDeps):
    """组装并编译状态图。"""

    def route(state: AgentState) -> dict:
        messages = [{"role": "user", "content": _INTENT_PROMPT.format(question=state["question"])}]
        return {"intent": parse_intent(deps.llm.chat(messages, temperature=0.0))}

    def chitchat(state: AgentState) -> dict:
        system = _CHITCHAT_SYSTEM.format(memory=Memory(state.get("facts", [])).render())
        messages = [{"role": "system", "content": system}]
        messages.extend(state.get("history", []))
        messages.append({"role": "user", "content": state["question"]})
        return {"reply": deps.llm.chat(messages, temperature=TEMPERATURE_CHITCHAT)}

    def sensitive(state: AgentState) -> dict:
        return {"reply": SENSITIVE_REFUSAL}

    def rewrite(state: AgentState) -> dict:
        """结合历史把最新输入改写成独立问题；无历史时原样透传（省一次 LLM 调用）。"""
        history = state.get("history", [])
        if not history:
            return {"rewritten": state["question"]}
        messages = [
            {
                "role": "user",
                "content": _REWRITE_PROMPT.format(
                    history=_render_history(history), question=state["question"]
                ),
            }
        ]
        return {"rewritten": deps.llm.chat(messages, temperature=0.0).strip()}

    def retrieve(state: AgentState) -> dict:
        query = state["rewritten"]
        vector = deps.embedder.embed([query])[0]
        candidates = deps.store.search(
            vector, top_k=deps.retrieve_top_k, filters=deps.search_filters
        )
        if deps.min_score > 0.0:
            candidates = [c for c in candidates if c.score >= deps.min_score]
        top = rerank(query, candidates, top_k=deps.rerank_top_k)
        return {"retrieved": top}

    def plan(state: AgentState) -> dict:
        """工具决策：无工具可调用时直接走检索（零额外 LLM 调用）。"""
        if not deps.tools:
            return {"tool_call": {}}
        catalog = "\n".join(
            f"- {t.name}：{t.description}\n  参数："
            + json.dumps(
                t.args_schema.model_json_schema().get("properties", {}), ensure_ascii=False
            )
            for t in deps.tools.values()
        )
        prompt = _PLAN_PROMPT.format(tool_catalog=catalog, question=state["rewritten"])
        decision = parse_plan(
            deps.llm.chat([{"role": "user", "content": prompt}], temperature=0.0),
            set(deps.tools),
        )
        if decision["action"] == "tool":
            return {"tool_call": {"tool": decision["tool"], "args": decision["args"]}}
        return {"tool_call": {}}

    def tool_exec(state: AgentState) -> dict:
        """执行 plan 决策的工具。参数校验失败、工具异常统一收敛为 ToolResult.error。"""
        call = state["tool_call"]
        tool = deps.tools[call["tool"]]
        try:
            args = tool.args_schema.model_validate(call["args"])
        except Exception as e:  # pydantic ValidationError；不向上抛
            return {"tool_result": ToolResult(error=f"工具参数非法: {e}")}
        ctx = deps.tool_ctx or QueryContext(session_id="agent", pack_name="")
        try:
            result = asyncio.run(tool.run(args, ctx))
        except Exception as e:
            return {"tool_result": ToolResult(error=f"工具执行异常: {e}")}
        return {"tool_result": result}

    def answer_from_tool(state: AgentState) -> dict:
        """依据工具输出作答。工具返回的 chunk 与检索 chunk 走同一套引用校验。"""
        result = state["tool_result"]
        if not result.ok:
            return {
                "answer": Answer(
                    text=f"抱歉，查询失败了：{result.error}",
                    citations=[],
                    confidence=0.9,
                    status="refused",
                    latency_ms=0,
                )
            }
        context = ""
        if result.data:
            context += "工具返回数据：\n" + json.dumps(result.data, ensure_ascii=False, indent=2)
        if result.chunks:
            context += "\n\n相关条文：\n" + build_context(result.chunks)
        # system 沿用领域模板（硬约束不变）；user 直接说明这是工具查询结果
        messages = [
            {"role": "system", "content": deps.template.system},
            {
                "role": "user",
                "content": f"{context}\n\n用户问题：{state['rewritten']}\n\n"
                "请仅根据以上工具查询结果回答；结果中没有的信息不要编造。",
            },
        ]
        start = time.perf_counter()
        text = deps.llm.chat(messages, temperature=TEMPERATURE_LEGAL)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = build_answer(text, result.chunks, latency_ms)
        if answer.status == "answered":
            answer.text += deps.disclaimer
        elif result.data and not result.chunks:
            # 数据类工具（如时效校验）没有可引用的 chunk：
            # 模型常对 law_id 也顺手加【】标注，这类"引用"没有 chunk 可核验，
            # 剥除标注后按数据回答放行，而不是误判为编造引用
            answer.text = strip_citation_markers(answer.text).strip() + deps.disclaimer
            answer.citations = []
            answer.status = "answered"
            answer.confidence = 0.8
        return {"answer": answer}

    def generate(state: AgentState) -> dict:
        chunks = [hit.chunk for hit in state["retrieved"]]
        if not chunks:
            # 空检索（如低于相关性阈值）→ 确定性拒答，不浪费 LLM 调用
            return {
                "answer": Answer(
                    text="抱歉，知识库中没有找到与您问题相关的资料，无法可靠回答。"
                    "您可以换个问法，或补充更多背景信息。",
                    citations=[],
                    confidence=0.9,
                    status="refused",
                    latency_ms=0,
                )
            }
        messages = build_messages(deps.template, chunks, state["rewritten"])
        start = time.perf_counter()
        text = deps.llm.chat(messages, temperature=TEMPERATURE_LEGAL)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = build_answer(text, chunks, latency_ms)
        if answer.status == "answered":
            answer.text += deps.disclaimer
        return {"answer": answer}

    def update_memory(state: AgentState) -> dict:
        """从本轮对话抽取长期事实。所有路径的收尾节点。"""
        reply = state.get("reply") or (state["answer"].text if state.get("answer") else "")
        existing = list(state.get("facts", []))
        new_facts = extract_facts(deps.llm, state["question"], reply, existing)
        return {"facts": existing + [f for f in new_facts if f not in existing]}

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("chitchat", chitchat)
    graph.add_node("sensitive", sensitive)
    graph.add_node("rewrite", rewrite)
    graph.add_node("plan", plan)
    graph.add_node("tool_exec", tool_exec)
    graph.add_node("answer_from_tool", answer_from_tool)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["intent"],
        {"chitchat": "chitchat", "legal": "rewrite", "sensitive": "sensitive"},
    )
    graph.add_edge("rewrite", "plan")
    graph.add_conditional_edges(
        "plan",
        lambda state: "tool" if state.get("tool_call") else "retrieve",
        {"tool": "tool_exec", "retrieve": "retrieve"},
    )
    graph.add_edge("tool_exec", "answer_from_tool")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("chitchat", "update_memory")
    graph.add_edge("sensitive", "update_memory")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("answer_from_tool", "update_memory")
    graph.add_edge("update_memory", END)
    return graph.compile()
