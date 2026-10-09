"""grounded prompt 组装。

模板由领域包提供（KnowledgePack.prompt_template()），引擎只负责渲染；
DEFAULT_TEMPLATE 为领域无关的兜底模板。模板中禁止硬编码任何领域措辞。
"""

from __future__ import annotations

from core.service.schemas import Chunk, PromptTemplate

# 领域无关兜底模板；领域包应提供自己的模板（如法律包的"引用条文必须现行有效"）
DEFAULT_TEMPLATE = PromptTemplate(
    system=(
        "你是一个严谨的知识库助手。规则：\n"
        "1. 只能依据给定资料回答，不得使用资料之外的知识；\n"
        "2. 回答中引用资料时，必须在对应句子末尾标注出处，格式为【chunk_id】，"
        "chunk_id 要完整照抄资料块标题中【】内的内容，不得简写、改写或编造；\n"
        "3. 资料不足以回答时，明确说明无法回答并解释原因，不得编造。"
    ),
    user="资料：\n{context}\n\n问题：{question}",
)


def build_context(chunks: list[Chunk]) -> str:
    """把召回的 chunks 渲染成带编号的资料块，供模板 {context} 占位符使用。"""
    blocks = []
    for chunk in chunks:
        path = " > ".join(chunk.metadata.section_path)
        header = f"【{chunk.chunk_id}】来源：{chunk.metadata.source}"
        if path:
            header += f"（{path}）"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(
    template: PromptTemplate,
    chunks: list[Chunk],
    question: str,
) -> list[dict[str, str]]:
    """组装 OpenAI chat 格式的 messages。"""
    return [
        {"role": "system", "content": template.system},
        {
            "role": "user",
            "content": template.render_user(context=build_context(chunks), question=question),
        },
    ]
