"""环境连通性检查：开发前一键确认 Qdrant / embedding / LLM 三个端点就绪。

用法：
    uv run python scripts/check_env.py
"""

from __future__ import annotations

import sys

import httpx

from core.config import get_settings


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    settings = get_settings()
    all_ok = True

    # Qdrant
    try:
        resp = httpx.get(f"{settings.qdrant_url.rstrip('/')}/collections", timeout=5.0)
        all_ok &= check("Qdrant", resp.status_code == 200, settings.qdrant_url)
    except httpx.HTTPError as e:
        all_ok &= check("Qdrant", False, f"{settings.qdrant_url} 连接失败: {e}")

    # Embedding（用一条探测文本实际请求一次）
    try:
        resp = httpx.post(
            f"{settings.embedding_base_url.rstrip('/')}/embeddings",
            json={"model": settings.embedding_model, "input": ["探测"]},
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            timeout=30.0,
        )
        dim = len(resp.json()["data"][0]["embedding"]) if resp.status_code == 200 else None
        all_ok &= check(
            "Embedding",
            resp.status_code == 200,
            f"{settings.embedding_model}（维度 {dim}）" if dim else resp.text[:100],
        )
    except httpx.HTTPError as e:
        all_ok &= check("Embedding", False, f"连接失败: {e}")

    # LLM
    try:
        resp = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": "回复“就绪”两个字即可"}],
                "max_tokens": 16,
            },
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=60.0,
        )
        all_ok &= check("LLM", resp.status_code == 200, settings.llm_model)
    except httpx.HTTPError as e:
        all_ok &= check("LLM", False, f"连接失败: {e}")

    print()
    if all_ok:
        print("环境就绪，可以跑 smoke.py")
        return 0
    print("有端点不可用：检查 .env 配置或服务器服务状态")
    return 1


if __name__ == "__main__":
    sys.exit(main())
