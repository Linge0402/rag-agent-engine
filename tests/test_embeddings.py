"""Embedding 客户端测试：httpx MockTransport 模拟端点，离线可跑。"""

import json

import httpx
import pytest

from core.indexing.embeddings import EmbeddingClient, OpenAIEmbeddingClient


def make_mock_transport(dim: int = 4) -> httpx.MockTransport:
    """模拟 /embeddings：返回与输入等长的向量列表，index 故意乱序。"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        inputs = payload["input"]
        # 乱序返回，验证客户端按 index 对齐输入顺序
        data = [
            {"index": i, "embedding": [float(i)] + [0.0] * (dim - 1)}
            for i in reversed(range(len(inputs)))
        ]
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


def make_client(batch_size: int = 2) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        base_url="http://fake-embedding/v1",
        model="bge-m3",
        batch_size=batch_size,
        client=httpx.Client(transport=make_mock_transport()),
    )


def test_embed_preserves_input_order():
    vectors = make_client().embed(["甲", "乙"])
    assert vectors[0][0] == 0.0  # 第一个输入对应 index 0
    assert vectors[1][0] == 1.0


def test_embed_batches_requests():
    client = make_client(batch_size=2)
    vectors = client.embed(["一", "二", "三"])  # 应分 2 批
    assert len(vectors) == 3
    # 第三批（单个）的 index 从 0 开始，验证按批内 index 对齐
    assert vectors[2][0] == 0.0


def test_embed_empty_input():
    assert make_client().embed([]) == []


def test_dim_probed_lazily():
    client = make_client()
    assert client._dim is None
    assert client.dim == 4
    assert client._dim == 4  # 缓存，不重复请求


def test_protocol_conformance():
    assert isinstance(make_client(), EmbeddingClient)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        make_client(batch_size=0)
