# rag-agent-engine

"引擎 + 领域包"架构的开源 Agentic RAG 项目：核心引擎（`core/`）沉淀可复用的解析、切分、检索、评估与 LangGraph 编排能力；领域包（`packs/`）以插件形式接入公开数据，当前规划两个实例：

- `packs/legal` —— 中国法律法规知识库 Agent（效力级别/时效状态硬过滤、时效校验、新旧法对比）
- `packs/devdocs` —— 开发者技术文档助手（库名+版本号过滤、版本兼容查询、代码沙箱）

## 文档

- 架构设计：`docs/architecture.md`
- 实施计划：`docs/implementation-plan.md`

## 状态

W1a —— 仓库骨架与核心抽象（`KnowledgePack` / `ChunkStrategy` / `Tool`）已就位，实现进行中。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest
```
