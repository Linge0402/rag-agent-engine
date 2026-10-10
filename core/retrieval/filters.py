"""声明式 metadata 硬过滤 DSL。

设计动机：企业知识库场景验证过的结论——过滤条件必须由领域包**显式声明**，
查询侧只能在声明范围内组合；否则任意字段都能当过滤条件，
typo（如 validity 写成 validty）会静默退化成"不过滤"，检索结果不可信。

工作方式：
- KnowledgePack.filter_schema() 声明允许过滤的字段与类型；
- 查询侧构造 FilterSet，validate() 时对照 schema 校验字段合法性与值类型；
- 校验通过后才翻译成向量库的等值条件字典（AND 组合）。

当前支持 op：eq（等值）、in（多值任一）。更复杂的组合（范围、否定）
在检索层出现真实需求前不引入——DSL 的能力边界宁窄勿滥。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from core.service.schemas import FilterSchema


class FilterCondition(BaseModel):
    """一条过滤条件。value 的类型须与 schema 声明的字段类型一致。"""

    field: str
    op: Literal["eq", "in"] = "eq"
    value: Any


class FilterSet(BaseModel):
    """一组 AND 组合的过滤条件。"""

    conditions: list[FilterCondition] = Field(default_factory=list)

    def validate(self, schema: FilterSchema) -> list[str]:
        """对照 pack 声明的 schema 校验，返回问题列表（空 = 合法）。"""
        problems: list[str] = []
        declared = {f.name: f.type for f in schema.fields}
        for cond in self.conditions:
            if cond.field not in declared:
                problems.append(f"字段 {cond.field!r} 未在 pack 的 filter_schema 中声明，禁止过滤")
                continue
            expected = declared[cond.field]
            values = cond.value if cond.op == "in" else [cond.value]
            if cond.op == "in" and not isinstance(cond.value, list):
                problems.append(f"op=in 的 value 必须是列表: {cond.field}")
                continue
            for v in values:
                if not _type_matches(v, expected):
                    problems.append(
                        f"字段 {cond.field!r} 声明类型为 {expected}，值 {v!r} 类型不符"
                    )
        return problems

    def to_store_conditions(self) -> dict[str, Any]:
        """翻译成向量库的等值条件字典（调用前须先通过 validate）。"""
        return {c.field: c.value for c in self.conditions}


def _type_matches(value: Any, declared: str) -> bool:
    return {
        "keyword": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, (int, float)) and not isinstance(value, bool),
        "bool": isinstance(value, bool),
        "datetime": isinstance(value, str),
    }.get(declared, False)
