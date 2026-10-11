"""对 OpenAI 兼容 chat 端点做并发压测：延迟分位数 + 吞吐。

用法：
    uv run python deploy/benchmark/bench.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
        --requests 32 --concurrency 8

默认读取 RAE_LLM_BASE_URL / RAE_LLM_MODEL / RAE_LLM_API_KEY（与 core/config.py 同一套约定），
所以不配参数时直接压本地 Ollama。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import httpx

# 覆盖短/中/长三种典型提问长度，避免单一 prompt 带来乐观估计
SAMPLE_PROMPTS = [
    "诉讼时效是多久？",
    "公司法规定的有限责任公司注册资本最低限额是多少？请给出法条依据。",
    "用人单位违法解除劳动合同，劳动者可以主张哪些权利？"
    "请分点说明，并标注对应的法律条文。",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI 兼容端点并发压测")
    p.add_argument("--base-url", default=os.environ.get("RAE_LLM_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--model", default=os.environ.get("RAE_LLM_MODEL", "qwen2.5:7b"))
    p.add_argument("--api-key", default=os.environ.get("RAE_LLM_API_KEY", "ollama"))
    p.add_argument("--requests", type=int, default=20, help="总请求数")
    p.add_argument("--concurrency", type=int, default=4, help="并发数")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


async def one_request(client: httpx.AsyncClient, args: argparse.Namespace, idx: int) -> dict:
    prompt = SAMPLE_PROMPTS[idx % len(SAMPLE_PROMPTS)]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    t0 = time.perf_counter()
    resp = await client.post(f"{args.base_url}/chat/completions", json=payload)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    return {
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def run(args: argparse.Namespace) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=300.0)
    headers = {"Authorization": f"Bearer {args.api_key}"}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:

        async def guarded(idx: int) -> dict:
            async with sem:
                return await one_request(client, args, idx)

        wall_t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall_t0

    latencies = sorted(r["latency"] for r in results)
    completion_tokens = sum(r["completion_tokens"] for r in results)

    def pct(q: float) -> float:
        # 线性插值分位数
        if len(latencies) == 1:
            return latencies[0]
        k = (len(latencies) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(latencies) - 1)
        return latencies[lo] + (latencies[hi] - latencies[lo]) * (k - lo)

    print(f"\n目标: {args.base_url}  模型: {args.model}")
    print(f"请求: {args.requests}  并发: {args.concurrency}  max_tokens: {args.max_tokens}")
    print("-" * 56)
    print(f"总耗时        : {wall:.1f} s")
    print(f"请求速率      : {args.requests / wall:.2f} req/s")
    print(f"输出吞吐      : {completion_tokens / wall:.1f} tok/s")
    print(f"延迟 mean     : {statistics.mean(latencies):.2f} s")
    print(f"延迟 p50      : {pct(0.50):.2f} s")
    print(f"延迟 p95      : {pct(0.95):.2f} s")
    print(f"延迟 max      : {latencies[-1]:.2f} s")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
