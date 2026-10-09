# rag-agent-engine 架构设计

> 一个"引擎 + 领域包"的开源 Agentic RAG 项目：核心引擎沉淀可复用的检索/评估/编排能力，
> 领域包（legal / devdocs）以插件形式接入公开数据，验证架构的复用性。
> 全部代码从零实现，不依赖任何历史项目代码。

## 1. 设计目标

1. **可复用**：领域无关的逻辑全部收进 `core/`，领域差异收敛到 `packs/` 的声明式配置与工具。
2. **可评估**：检索与生成指标是一等公民，任何策略变更都能跑出对比数据。
3. **可追溯**：每个回答带条文/文档级引用，低置信度拒答，拒绝"无依据生成"。
4. **可演示**：`docker compose up` 一键起两个 demo，README 放 GIF 和评估数据。

## 2. 目录结构

```
rag-agent-engine/
├── core/                        # 领域无关引擎
│   ├── parsing/                 # 多格式文档解析
│   │   ├── loaders.py           #   md/pdf/docx/doc/excel 解析（含 ZIP entry/解压大小/图片数安全限制）
│   │   ├── ocr.py               #   扫描件 OCR（PaddleOCR 延迟加载 + 像素预算校验）
│   │   └── cleaning.py          #   页眉页脚/空白噪声清洗
│   ├── chunking/
│   │   ├── base.py              #   ChunkStrategy 抽象：切分 + metadata 抽取
│   │   └── structural.py        #   标题/段落结构切分（默认实现）
│   ├── indexing/
│   │   ├── embeddings.py        #   bge-m3 批量向量化
│   │   ├── vector_store.py      #   Qdrant 封装：collection/幂等写入/过滤检索
│   │   └── ingestion.py         #   入库编排：分流→解析→切分→向量化→写入
│   ├── retrieval/
│   │   ├── filters.py           #   metadata 硬过滤 DSL（项目卖点，声明式设计）
│   │   └── rerank.py            #   向量分 + 词项重合度轻量重排
│   ├── generation/
│   │   ├── prompt.py            #   grounded prompt 模板（模板 per-pack 配置化，不硬编码领域措辞）
│   │   └── llm.py               #   OpenAI-compatible 客户端 + 引用校验
│   ├── evaluation/
│   │   ├── retrieval_metrics.py #   top-k 召回 / MRR
│   │   └── answer_metrics.py    #   引用准确率 / 拒答正确率
│   ├── agent/                   # LangGraph 编排层
│   │   ├── graph.py             #   状态图：改写→检索→评估→生成→自检
│   │   ├── tools.py             #   工具协议 + 注册表（search / lookup / compare ...）
│   │   └── memory.py            #   多轮对话状态管理
│   ├── mcp/
│   │   └── server.py            # 把检索能力暴露为 MCP Server
│   ├── service/
│   │   ├── api.py               #   FastAPI 接口
│   │   └── schemas.py           #   数据契约
│   ├── pack.py                  # KnowledgePack 协议（领域包接入的核心抽象）
│   └── config.py                # 集中配置，全部走环境变量
├── packs/
│   ├── legal/                   # 领域包 A：中国法律法规
│   │   ├── pack.py              #   KnowledgePack 声明：解析器/切分/元数据/工具/提示词
│   │   ├── parsers.py           #   法规"编-章-节-条"结构解析
│   │   ├── tools.py             #   时效校验、新旧法对比、条文定位
│   │   ├── prompts.py           #   法律问答 grounded 模板（含"现行有效"约束）
│   │   └── eval/                #   评估集：法条定位/时效判断/边界问题
│   └── devdocs/                 # 领域包 B：开发者技术文档
│       ├── pack.py
│       ├── parsers.py           #   官方文档 HTML/Markdown 导入 + 版本 metadata
│       ├── tools.py             #   版本兼容查询、代码片段沙箱执行
│       ├── prompts.py
│       └── eval/
├── deploy/
│   ├── docker-compose.yml       #   Qdrant + vLLM(Qwen2.5-7B) + app
│   └── benchmark/               #   并发/延迟压测脚本 + 报告
├── scripts/
│   └── smoke.py                 #   端到端冒烟脚本
├── tests/                       # 契约测试 + 模块单测，示例数据一律用公开领域语料
└── README.md
```

## 3. 核心接口契约

### 3.1 数据契约

```python
class Chunk(BaseModel):
    chunk_id: str                 # 可追溯 ID：[doc-001-p3-c2]
    doc_id: str
    text: str
    metadata: ChunkMetadata       # source/section_path/page/version/validity/ACL...

class Answer(BaseModel):
    text: str
    citations: list[Citation]     # 每条结论挂 chunk_id + 原文片段
    confidence: float
    status: Literal["answered", "refused", "needs_review"]
    latency_ms: int
```

### 3.2 领域包协议（本项目最核心的抽象）

```python
class KnowledgePack(Protocol):
    name: str
    def parsers(self) -> list[Parser]: ...          # 领域文档解析
    def chunk_strategy(self) -> ChunkStrategy: ...  # 领域切分策略
    def filter_schema(self) -> FilterSchema: ...    # 可过滤字段声明（如"现行有效"）
    def tools(self) -> list[Tool]: ...              # 暴露给 Agent 的领域工具
    def prompt_template(self) -> PromptTemplate: ...# 领域 grounded 模板
    def eval_sets(self) -> list[EvalSet]: ...       # 领域评估集
```

### 3.3 Agent 工具协议

```python
class Tool(BaseModel):
    name: str
    description: str              # 供 LLM 决策调用
    args_schema: type[BaseModel]
    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult: ...
```

## 4. 数据流

### 4.1 入库流水线

```
文件 → 类型分流 → 解析(文本层优先/OCR兜底) → 清洗 → 领域切分策略
     → metadata 抽取 → bge-m3 向量化 → Qdrant 幂等写入（带版本/ACL）
```

### 4.2 查询流水线（分两期演进）

```
【v1 基线-单轮检索】query → embed → metadata 硬过滤 → top-20 召回
              → 轻量重排 top-6 → grounded prompt → LLM → 引用校验 → 返回

【v2-Agent 编排】query → [LangGraph 状态图]
  ├─ 改写节点：多轮上下文压缩 + 查询改写
  ├─ 检索节点：metadata 硬过滤 + 召回 + 重排（复用 v1 组件）
  ├─ 评估节点：相关性自检，不足则改写重试（≤2 次）
  ├─ 工具节点：领域工具调用（时效校验/版本查询/沙箱）
  ├─ 生成节点：grounded prompt + 引用约束（复用 v1 组件）
  └─ 自检节点：引用合法性校验，非法引用 → 拒答或复核
```

### 4.3 MCP 暴露面

```
MCP Server tools:
  - knowledge_search(query, filters)   → 任意 Agent 可调用本知识库
  - get_chunk(chunk_id)                → 取引用原文
  - legal_check_validity(law, article) → 领域包 A 专用工具
```

## 5. 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 向量库 | Qdrant | 生态成熟、过滤检索能力强，docker compose 一键起 |
| Embedding | bge-m3 | 中文检索效果好，可完全本地化部署 |
| 编排 | LangGraph | 状态图适合"检索-自检-重试"循环，也是本项目要展示的核心能力 |
| LLM 服务 | vLLM + Qwen2.5-7B（OpenAI-compatible） | 接口标准化、本地/云端可切换；顺带产出部署与压测章节 |
| 领域差异 | KnowledgePack 协议 | "引擎+领域包"是项目的主要卖点 |
| 评估 | 离线评估集 + 报告脚本 | 面试官最看重的差异化，必须前置设计而非事后补 |
| 开发期依赖 | 任意 OpenAI 兼容端点 / Ollama 本地小模型 | vLLM 部署留到 W6，不阻塞链路开发 |

## 6. 里程碑

| 周 | 交付物 | 验收标准 |
|---|---|---|
| W1 | core 骨架 + 最小端到端链路 | 三个核心抽象定稿；一条 query 从入库到带引用回答跑通；测试全绿 |
| W2 | core 引擎完整实现 | 全格式解析、过滤 DSL、评估指标齐备；示例数据全部来自公开领域语料 |
| W3-4 | packs/legal + Agent 编排 | 法规 demo 可问答；时效工具有单测；评估集跑出基线报告 |
| W5 | packs/devdocs | 只写 pack 声明与 2 个工具，验证"不改 core 即可接入新领域" |
| W6 | deploy + README | docker compose 一键起；压测报告；两个 GIF demo |
