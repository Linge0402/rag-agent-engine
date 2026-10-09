# 实施计划：rag-agent-engine 从零构建

> 本仓库全部代码从零实现，不迁移、不复制任何历史项目代码。
> `architecture.md` 是架构与接口的唯一权威来源；本文件规定构建顺序、模块规格与验收标准。
> 历史项目中验证过的设计结论（ZIP/解压安全限制、OCR 像素预算、幂等写入、Fake 组件测试法等）
> 作为**规格要求**保留在下方模块清单中，而非作为代码保留。

## 一、实现原则

1. **接口先行**：先定 `KnowledgePack` / `ChunkStrategy` / `Tool` 三个抽象并评审，再写任何实现。
   接口定错，后面全部返工。
2. **测试随模块交付**：每个模块至少 1 个单测；外部依赖（embedding/向量库/LLM）一律用
   Fake 组件注入，保证测试离线可跑。
3. **示例数据公开化**：测试夹具、冒烟脚本、demo 语料一律使用公开领域文本
   （法规条文、开源项目文档），从源头避免任何脱敏问题。
4. **开发期轻依赖**：LLM 走任意 OpenAI 兼容端点或 Ollama 本地小模型，embedding 可先用
   本地 bge-m3 或 Fake；vLLM + Qwen2.5-7B 部署留到 W6，不阻塞链路开发。
5. **git 从零提交**：`git init` 全新历史，语义化 commit，每周一个可运行状态。

## 二、构建顺序与模块规格

### W1a：仓库骨架 + 核心抽象（约 1 天）

| 交付物 | 规格要点 |
|---|---|
| 工程化骨架 | `pyproject.toml`（uv 管理）、`ruff`、`pytest`、GitHub Actions（lint + test）、`.gitignore` |
| `core/service/schemas.py` | `Chunk` / `ChunkMetadata` / `Citation` / `Answer` 契约（见 architecture.md §3.1） |
| `core/chunking/base.py` | `ChunkStrategy` 抽象：输入解析后文档，输出带 metadata 的 `Chunk` 序列 |
| `core/pack.py` | **`KnowledgePack` 协议——整个项目的心脏**，接入新领域需要声明的全部内容都在这（§3.2） |
| `core/config.py` | 集中配置，全部走环境变量，pydantic-settings |

**验收**：三个抽象经评审定稿；`pytest` 可跑（至少含契约 schema 校验测试）；`ruff` 通过；完成首次 commit。

### W1b：最小端到端链路（约 3～5 天）

只实现"一条 query 从入库到出答案"的最短路径，验证语料用 LangGraph 官方文档（公开、结构简单）：

| 模块 | 本阶段最小实现 | 完整规格（W2 补齐） |
|---|---|---|
| `core/parsing/loaders.py` | 仅 Markdown | 补 pdf/docx/doc/excel；ZIP entry 数、解压大小、图片数安全上限 |
| `core/chunking/structural.py` | 按标题层级 + 段落切分 | 实现 `ChunkStrategy` 默认策略，metadata 含 section_path |
| `core/indexing/embeddings.py` | bge-m3 批量封装（本地或兼容端点） | — |
| `core/indexing/vector_store.py` | Qdrant collection 管理 + 幂等写入 + 过滤检索 | 过滤逻辑将下沉到 filters.py |
| `core/indexing/ingestion.py` | 分流→解析→切分→向量化→写入 | 切分策略改为从 KnowledgePack 注入 |
| `core/retrieval/rerank.py` | 向量分 + 词项重合度组合排序 | — |
| `core/generation/prompt.py` | grounded 模板，**模板从 pack 配置注入，禁止硬编码领域措辞** | — |
| `core/generation/llm.py` | OpenAI-compatible 客户端 + 引用标记校验 | — |
| `scripts/smoke.py` | 端到端冒烟：入库 2~3 篇文档 → 提问 → 带引用回答 | — |

**验收**：提问 "LangGraph 的 StateGraph 怎么定义节点" 能返回带 `chunk_id` 引用的答案；各模块单测齐备。

### W2：引擎完善

| 模块 | 规格要点 |
|---|---|
| `core/parsing/loaders.py` 补全 | pdf（文本层优先）/docx/doc/excel；安全限制齐备 |
| `core/parsing/ocr.py` | PaddleOCR 延迟加载；扫描件像素预算校验，防爆内存 |
| `core/parsing/cleaning.py` | 页眉页脚/空白噪声清洗 |
| `core/retrieval/filters.py` | **声明式 metadata 硬过滤 DSL**（项目卖点，要放大设计）：从 vector_store 拆出，pack 声明可过滤字段，查询侧组合条件 |
| `core/evaluation/retrieval_metrics.py` | top-k 召回命中率 + MRR |
| `core/evaluation/answer_metrics.py` | 引用准确率、拒答正确率 |
| `core/service/api.py` | FastAPI 查询/入库接口，查询编排先保持 v1 单轮链路 |

**验收**：架构文档里程碑 W2 标准——测试全绿，示例数据全部公开语料。

### W3-4：packs/legal + Agent 编排（本项目核心增量）

| 模块 | 规格要点 |
|---|---|
| `core/agent/graph.py` | LangGraph 状态图：改写→检索→评估→生成→自检；检索不足改写重试 ≤2 次 |
| `core/agent/tools.py` + `memory.py` | 工具协议/注册表 + 多轮对话状态 |
| `core/mcp/server.py` | MCP Server 暴露 `knowledge_search` / `get_chunk` |
| `packs/legal/parsers.py` | 法规"编-章-节-条"层级解析，metadata：效力级别/时效状态/发文机关/施行日期 |
| `packs/legal/tools.py` | 时效校验、新旧法对比、条文定位（每个工具配单测） |
| `packs/legal/prompts.py` | 法律问答 grounded 模板，含"引用条文必须现行有效"硬约束 |
| `packs/legal/eval/` | 评估集：法条定位 / 时效判断 / 边界问题，跑出基线报告 |
| 数据源 | 国家法律法规数据库（flk.npc.gov.cn）等公开渠道 |

### W5：packs/devdocs（验证架构复用性）

只写数据接入和 2 个工具，**不改 core 一行代码**——这本身就是验收标准：

| 模块 | 规格要点 |
|---|---|
| `packs/devdocs/parsers.py` | 官方文档 HTML/Markdown 导入，metadata：库名 + 版本号 |
| `packs/devdocs/tools.py` | 版本兼容查询、代码片段沙箱执行 |

### W6：部署 + 收尾

| 交付物 | 规格要点 |
|---|---|
| `deploy/docker-compose.yml` | Qdrant + vLLM(Qwen2.5-7B) + app，一键起两个 demo |
| `deploy/benchmark/` | 并发/延迟压测脚本 + 报告数据 |
| `README.md` | 架构图、两个 demo GIF、评估数据、快速开始 |

## 三、每周检查点

每周末对照 architecture.md §6 里程碑验收；未达标项进入下周计划，不带着债务进入下一阶段。
