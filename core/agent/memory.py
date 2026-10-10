"""多轮对话记忆：窗口管对话流，事实抽取管长期记忆。

设计背景：固定窗口（如最近 5 轮）成本可控，但早期信息会被淘汰——
用户第 1 轮说的名字，第 6 轮就"忘记"了。
本模块的解法：每轮对话后用 LLM 抽取**长期有效的事实**（姓名、身份、
案件背景、偏好）存入 Memory，注入后续 prompt；闲聊细节仍随窗口淘汰。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.generation.llm import LLMClient

_EXTRACT_PROMPT = """从下面这轮对话中提取关于用户的、长期有效的事实
（如姓名、身份、职业、案件背景、明确偏好）。
规则：
- 只提取用户明确陈述的事实，禁止推测；
- 每行一条，用简短陈述句；
- 不要重复已知事实；
- 没有新事实时，只回复一个字：无

已知事实：
{existing}

本轮对话：
用户：{question}
助手：{reply}"""


@dataclass
class Memory:
    """跨轮持久化的用户事实档案。"""

    facts: list[str] = field(default_factory=list)

    def render(self) -> str:
        """渲染成 prompt 片段；无事实时返回空串（不污染 prompt）。"""
        if not self.facts:
            return ""
        return "关于用户的已知信息：\n" + "\n".join(f"- {f}" for f in self.facts)

    def add(self, new_facts: list[str]) -> None:
        for fact in new_facts:
            if fact not in self.facts:
                self.facts.append(fact)


def extract_facts(
    llm: LLMClient,
    question: str,
    reply: str,
    existing: list[str],
) -> list[str]:
    """从一轮对话中抽取新事实。LLM 输出"无"或无可解析内容时返回空列表。"""
    prompt = _EXTRACT_PROMPT.format(
        existing="\n".join(f"- {f}" for f in existing) or "（暂无）",
        question=question,
        reply=reply,
    )
    text = llm.chat([{"role": "user", "content": prompt}])
    if text.strip().startswith("无"):
        return []

    facts: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*、0123456789. ").strip()
        if cleaned and cleaned != "无" and cleaned not in existing:
            facts.append(cleaned)
    return facts
