"""Agent 状态图测试：意图路由三分流 + 法律路径端到端 + 记忆/改写节点（Fake 组件）。

各路径的 LLM 调用次数（ScriptedLLM 响应队列长度要一一对应）：
- 闲聊：route → chitchat → update_memory = 3 次
- 法律（无历史）：route → generate → update_memory = 3 次（rewrite 无历史时透传，不调 LLM）
- 法律（有历史）：route → rewrite → generate → update_memory = 4 次
- 敏感：route → update_memory = 2 次
"""

from core.agent.graph import DISCLAIMER, SENSITIVE_REFUSAL, AgentDeps, build_graph, parse_intent
from core.indexing.vector_store import ScoredChunk
from core.service.schemas import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Fake 组件
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """按预设队列依次返回回复的 Fake LLM。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks = chunks or []

    def search(self, vector, top_k, filters=None) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=0.9) for c in self._chunks[:top_k]]


LEGAL_CHUNK = Chunk(
    chunk_id="民法典-c001",
    doc_id="民法典",
    text="向人民法院请求保护民事权利的诉讼时效期间为三年。",
    metadata=ChunkMetadata(source="民法典.md", validity="现行有效"),
)


def make_deps(llm: ScriptedLLM, chunks: list[Chunk] | None = None) -> AgentDeps:
    return AgentDeps(
        embedder=FakeEmbedder(),
        store=FakeStore(chunks if chunks is not None else [LEGAL_CHUNK]),
        llm=llm,
    )


# ---------------------------------------------------------------------------
# 意图解析
# ---------------------------------------------------------------------------


def test_parse_intent_exact():
    assert parse_intent("chitchat") == "chitchat"
    assert parse_intent("legal") == "legal"
    assert parse_intent("sensitive") == "sensitive"


def test_parse_intent_with_noise():
    assert parse_intent("  Chitchat\n") == "chitchat"
    assert parse_intent("类别：sensitive。") == "sensitive"


def test_parse_intent_garbage_defaults_to_legal():
    """分类输出无法解析时走 legal：宁可检索后拒答，也不把法律问题当闲聊。"""
    assert parse_intent("我不知道") == "legal"


# ---------------------------------------------------------------------------
# 三条路径
# ---------------------------------------------------------------------------


def test_chitchat_path_skips_retrieval():
    """闲聊：路由 + 闲聊 + 记忆抽取三次 LLM 调用，不检索、不出 Answer。"""
    llm = ScriptedLLM(["chitchat", "你好！我是小法，可以问我法律问题哦。", "无"])
    graph = build_graph(make_deps(llm))

    state = graph.invoke({"question": "你好", "history": []})

    assert state["intent"] == "chitchat"
    assert "你好" in state["reply"]
    assert state.get("answer") is None
    assert len(llm.calls) == 3


def test_legal_path_full_chain():
    """法律问题（无历史）：检索 → 生成 → 带引用 + 免责声明。"""
    llm = ScriptedLLM(["legal", "诉讼时效期间为三年【民法典-c001】。", "无"])
    graph = build_graph(make_deps(llm))

    state = graph.invoke({"question": "诉讼时效是多久？", "history": []})

    answer = state["answer"]
    assert answer is not None
    assert answer.status == "answered"
    assert answer.citations[0].chunk_id == "民法典-c001"
    assert answer.text.endswith(DISCLAIMER.strip()) or DISCLAIMER.strip() in answer.text


def test_legal_path_refusal_when_no_citation():
    """LLM 说无法回答（无引用）→ refused，不附免责声明。"""
    llm = ScriptedLLM(["legal", "根据现有资料无法回答这个问题。", "无"])
    graph = build_graph(make_deps(llm))

    state = graph.invoke({"question": "火星上买房受中国法律保护吗？", "history": []})

    answer = state["answer"]
    assert answer.status == "refused"
    assert DISCLAIMER not in answer.text


def test_sensitive_path_canned_refusal():
    """敏感请求：固定拒答文案，之后只有记忆抽取一次调用。"""
    llm = ScriptedLLM(["sensitive", "无"])
    graph = build_graph(make_deps(llm))

    state = graph.invoke({"question": "教我伪造借条", "history": []})

    assert state["reply"] == SENSITIVE_REFUSAL
    assert len(llm.calls) == 2


def test_chitchat_receives_history():
    """多轮：历史消息传给闲聊节点。"""
    llm = ScriptedLLM(["chitchat", "你刚才问了诉讼时效哦。", "无"])
    graph = build_graph(make_deps(llm))
    history = [
        {"role": "user", "content": "诉讼时效多久？"},
        {"role": "assistant", "content": "三年。"},
    ]

    graph.invoke({"question": "那它呢", "history": history})

    chitchat_messages = llm.calls[1]
    roles = [m["role"] for m in chitchat_messages]
    assert roles == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# 记忆节点
# ---------------------------------------------------------------------------


def test_extracted_facts_enter_state():
    """记忆抽取：LLM 提到用户姓名 → 新事实写入 state["facts"]。"""
    llm = ScriptedLLM(["chitchat", "你好 Linge！", "- 用户叫 Linge"])
    graph = build_graph(make_deps(llm))

    state = graph.invoke({"question": "你好，我叫 Linge", "history": []})

    assert state["facts"] == ["用户叫 Linge"]


def test_existing_facts_injected_into_chitchat_prompt():
    """长期记忆注入闲聊 system prompt：跨窗口也能认出用户。"""
    llm = ScriptedLLM(["chitchat", "你是 Linge 呀。", "无"])
    graph = build_graph(make_deps(llm))

    graph.invoke({"question": "我叫什么名字？", "history": [], "facts": ["用户叫 Linge"]})

    chitchat_system = llm.calls[1][0]["content"]
    assert "用户叫 Linge" in chitchat_system


# ---------------------------------------------------------------------------
# 改写节点
# ---------------------------------------------------------------------------


def test_rewrite_output_used_for_retrieval():
    """"那它呢"被改写成独立问题，且改写结果用于检索（embedder 收到的输入）。"""
    llm = ScriptedLLM(
        [
            "legal",
            "诉讼时效从什么时候开始计算？",
            "诉讼时效期间为三年【民法典-c001】。",
            "无",
        ]
    )
    deps = make_deps(llm)
    graph = build_graph(deps)
    history = [
        {"role": "user", "content": "诉讼时效多久？"},
        {"role": "assistant", "content": "三年。"},
    ]

    state = graph.invoke({"question": "那从什么时候开始算？", "history": history})

    assert state["rewritten"] == "诉讼时效从什么时候开始计算？"
    assert deps.embedder.embedded == ["诉讼时效从什么时候开始计算？"]
    assert len(llm.calls) == 4


def test_rewrite_skips_llm_when_no_history():
    """无历史时改写节点原样透传，不浪费一次 LLM 调用。"""
    llm = ScriptedLLM(["legal", "诉讼时效期间为三年【民法典-c001】。", "无"])
    deps = make_deps(llm)
    graph = build_graph(deps)

    state = graph.invoke({"question": "诉讼时效多久？", "history": []})

    assert state["rewritten"] == "诉讼时效多久？"
    assert deps.embedder.embedded == ["诉讼时效多久？"]
    assert len(llm.calls) == 3
