"""向量库测试：point id 确定性、payload 寻址、过滤条件翻译。

不连真实 Qdrant：只测纯逻辑函数与参数校验；
端到端行为由 scripts/smoke.py 覆盖。
"""

import uuid

from qdrant_client import models

from core.indexing.vector_store import build_filter, payload_key, point_id


def test_point_id_deterministic_and_valid_uuid():
    id1 = point_id("minfadian-c001")
    id2 = point_id("minfadian-c001")
    assert id1 == id2
    uuid.UUID(id1)  # 是合法 UUID
    assert id1 != point_id("minfadian-c002")


def test_payload_key_builtin_vs_extra():
    assert payload_key("validity") == "metadata.validity"
    assert payload_key("source") == "metadata.source"
    # 领域扩展字段（如法规的效力级别）走 extra 命名空间
    assert payload_key("效力级别") == "metadata.extra.效力级别"


def test_build_filter_translates_to_and_conditions():
    flt = build_filter({"validity": "现行有效", "效力级别": "法律"})
    assert flt is not None
    assert len(flt.must) == 2
    keys = {c.key for c in flt.must}
    assert keys == {"metadata.validity", "metadata.extra.效力级别"}


def test_build_filter_empty_returns_none():
    assert build_filter({}) is None


def test_build_filter_value_types():
    flt = build_filter({"page": 3})
    condition = flt.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.match.value == 3
