"""长期记忆模块测试：事实抽取解析 + Memory 渲染。"""

from core.agent.memory import Memory, extract_facts


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        self.calls.append(messages)
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Memory.render
# ---------------------------------------------------------------------------


def test_render_empty():
    assert Memory().render() == ""


def test_render_with_facts():
    memory = Memory(facts=["用户叫 Linge", "用户是工程师"])
    rendered = memory.render()
    assert rendered.startswith("关于用户的已知信息：")
    assert "- 用户叫 Linge" in rendered
    assert "- 用户是工程师" in rendered


def test_add_dedup():
    memory = Memory(facts=["用户叫 Linge"])
    memory.add(["用户叫 Linge", "用户是工程师"])
    assert memory.facts == ["用户叫 Linge", "用户是工程师"]


# ---------------------------------------------------------------------------
# extract_facts
# ---------------------------------------------------------------------------


def test_extract_returns_empty_when_llm_says_none():
    llm = ScriptedLLM(["无"])
    assert extract_facts(llm, "你好", "你好！", []) == []
    assert extract_facts(ScriptedLLM(["无。"]), "你好", "你好！", []) == []


def test_extract_parses_bullet_lines():
    llm = ScriptedLLM(["- 用户叫 Linge\n- 用户是工程师"])
    facts = extract_facts(llm, "我叫 Linge，是工程师", "好的 Linge！", [])
    assert facts == ["用户叫 Linge", "用户是工程师"]


def test_extract_strips_numbering_and_symbols():
    llm = ScriptedLLM(["1. 用户叫 Linge\n* 用户是工程师\n• 用户在做法律问答项目"])
    facts = extract_facts(llm, "...", "...", [])
    assert facts == ["用户叫 Linge", "用户是工程师", "用户在做法律问答项目"]


def test_extract_skips_duplicates_of_existing():
    llm = ScriptedLLM(["- 用户叫 Linge\n- 用户是工程师"])
    facts = extract_facts(llm, "...", "...", existing=["用户叫 Linge"])
    assert facts == ["用户是工程师"]


def test_extract_prompt_contains_context():
    """抽取 prompt 要带上已知事实和本轮对话（供 LLM 判断增量）。"""
    llm = ScriptedLLM(["无"])
    extract_facts(llm, "诉讼时效多久？", "三年。", existing=["用户叫 Linge"])
    prompt = llm.calls[0][0]["content"]
    assert "用户叫 Linge" in prompt
    assert "诉讼时效多久？" in prompt
    assert "三年。" in prompt
