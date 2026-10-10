"""过滤 DSL 测试：字段合法性校验是核心价值（typo 不能静默退化成不过滤）。"""

from core.retrieval.filters import FilterCondition, FilterSet
from core.service.schemas import FilterField, FilterSchema

SCHEMA = FilterSchema(
    fields=[
        FilterField(name="validity", type="keyword"),
        FilterField(name="extra.article_no", type="integer"),
    ]
)


def test_valid_eq_filter():
    fs = FilterSet(conditions=[FilterCondition(field="validity", value="现行有效")])
    assert fs.validate(SCHEMA) == []
    assert fs.to_store_conditions() == {"validity": "现行有效"}


def test_undeclared_field_rejected():
    """typo（validty）必须报错，不能静默放行。"""
    fs = FilterSet(conditions=[FilterCondition(field="validty", value="现行有效")])
    problems = fs.validate(SCHEMA)
    assert len(problems) == 1
    assert "validty" in problems[0]


def test_type_mismatch_rejected():
    fs = FilterSet(conditions=[FilterCondition(field="extra.article_no", value="一八八")])
    assert fs.validate(SCHEMA) != []


def test_in_op_requires_list():
    fs = FilterSet(conditions=[FilterCondition(field="validity", op="in", value="现行有效")])
    assert fs.validate(SCHEMA) != []
    ok = FilterSet(
        conditions=[FilterCondition(field="validity", op="in", value=["现行有效", "已修订"])]
    )
    assert ok.validate(SCHEMA) == []
    assert ok.to_store_conditions() == {"validity": ["现行有效", "已修订"]}


def test_bool_not_accepted_as_integer():
    """Python 里 bool 是 int 子类，但过滤语义上不允许混用。"""
    fs = FilterSet(conditions=[FilterCondition(field="extra.article_no", value=True)])
    assert fs.validate(SCHEMA) != []
