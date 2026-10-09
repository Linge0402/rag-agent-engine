"""集中配置：全部走环境变量（前缀 RAE_），支持 .env。

开发期默认指向本地轻依赖（Ollama / 本地 Qdrant），
生产部署（W6 vLLM）只改环境变量，不改代码。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAE_", env_file=".env", extra="ignore")

    # 向量库
    qdrant_url: str = "http://localhost:6333"

    # Embedding（OpenAI 兼容端点或本地推理）
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "bge-m3"
    embedding_api_key: str = "EMPTY"

    # LLM（OpenAI 兼容端点；开发期 Ollama，部署期 vLLM）
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = "EMPTY"

    # 检索默认值
    retrieve_top_k: int = 20  # 召回
    rerank_top_k: int = 6  # 重排后送入生成

    # 安全限制（解析阶段）
    max_zip_entries: int = 1000
    max_uncompressed_mb: int = 200
    max_images_per_doc: int = 500
    ocr_max_pixels: int = 40_000_000  # OCR 像素预算，防爆内存


def get_settings() -> Settings:
    return Settings()
