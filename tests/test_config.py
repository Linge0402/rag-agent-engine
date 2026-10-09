"""配置测试：默认值与 RAE_ 环境变量覆盖。"""

from core.config import Settings


def test_defaults_point_to_local_dev():
    settings = Settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_model == "bge-m3"
    assert settings.retrieve_top_k == 20
    assert settings.rerank_top_k == 6


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAE_LLM_BASE_URL", "http://vllm.internal:8000/v1")
    monkeypatch.setenv("RAE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("RAE_RETRIEVE_TOP_K", "50")

    settings = Settings()
    assert settings.llm_base_url == "http://vllm.internal:8000/v1"
    assert settings.llm_model == "Qwen/Qwen2.5-7B-Instruct"
    assert settings.retrieve_top_k == 50


def test_safety_limits_have_defaults():
    settings = Settings()
    assert settings.max_zip_entries > 0
    assert settings.max_uncompressed_mb > 0
    assert settings.ocr_max_pixels > 0
