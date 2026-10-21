# SmartMall Agent

面向智能电商客服场景的 **RAG + 业务 Agent**：既能基于知识库回答商品与售后问题，
也能在模拟业务环境中查商品、查订单、查物流、做退货资格预检查和创建售后工单。

项目的工程重点不是"接个大模型"，而是：**混合检索与可复现评测**、
**服务端强制的权限与业务规则**、**引用可追溯**、**Prompt Injection 与越权防护**。

## 核心特性

| 能力 | 说明 | 代码位置 |
|------|------|----------|
| 知识库构建 | Markdown + frontmatter → 按标题切块 → 嵌入 → Chroma；**增量更新幂等** | `src/sma/knowledge/` |
| 混合检索 | Vector / BM25(jieba 中文分词) / RRF 融合 / 轻量重排，全过程留痕 | `src/sma/rag/` |
| 回答生成 | 基于证据回答；无依据拒答；**引用真实性校验** | `src/sma/agent/validator.py` |
| 相关性门槛 | 召回内容与问题覆盖率不足时直接拒答，而不是让模型"基于上下文"编 | `src/sma/agent/guards.py` |
| 业务工具 | 商品/订单/物流/退货资格/售后创建/退款（高风险拦截），统一校验+超时+幂等 | `src/sma/tools/` |
| 权限与规则 | 订单归属校验、退货资格由**代码规则**判定、高风险操作模型不得执行 | `src/sma/tools/business_rules.py` |
| 注入防护 | 确定性模式匹配短路 + 提示词约束 + 能力最小化 + 输出校验 | `src/sma/agent/guards.py` |
| Agent 工作流 | LangGraph：意图路由 → 检索/工具/澄清/转人工 → 校验，无回边 | `src/sma/agent/workflow.py` |
| 评测系统 | 24 条标注用例，4 种检索模式对比，指标现场跑出不预填 | `src/sma/eval/` |
| 会话隔离 | 会话绑定 user_id，跨用户读取直接 403；历史有上限 | `src/sma/persistence/` |

## 技术栈

Python 3.11 / uv / FastAPI / Pydantic v2 / LangGraph / Chroma / jieba / httpx / aiosqlite /
pytest / Ruff / Docker / GitHub Actions。DeepSeek 作为默认模型提供方（OpenAI 兼容）。

**没有使用 CrewAI**：本项目是单 Agent + 条件路由，不存在多智能体协作需求，
引入 CrewAI 只会增加一层无收益的抽象和维护成本（见 [TECH_DECISIONS](docs/TECH_DECISIONS.md)）。

## 快速开始（全新环境，5 分钟，零密钥）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # 安装 uv

git clone <repo> && cd smartmall-agent
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env

uv run pytest                                        # 174 个离线测试，无需任何密钥
uv run sma index                                     # 构建知识库
uv run sma ask "7 天无理由退货从什么时候开始算"          # 知识问答
uv run sma ask "查一下订单 SO20261001001 的状态" --user U001   # 工具调用
uv run sma eval                                      # 跑评测集，输出真实指标
uv run sma serve                                     # http://127.0.0.1:8000/docs
```

Docker：

```bash
docker compose up --build
curl http://127.0.0.1:8000/health
```

## 配置

完整配置见 [.env.example](.env.example)，环境变量前缀 `SMA_`。关键项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SMA_LLM_PROVIDER` | `mock` | `mock` / `deepseek` |
| `SMA_LLM_API_KEY` | 空 | deepseek 模式必填 |
| `SMA_EMBEDDING_PROVIDER` | `hash` | `hash`（本地确定性，**词面相似非语义**）/ `deepseek_compatible` |
| `SMA_RETRIEVAL_MODE` | `hybrid` | `vector` / `bm25` / `hybrid` / `hybrid_rerank` |
| `SMA_TOP_K` | 4 | 最终进入上下文的块数 |
| `SMA_CANDIDATE_K` | 10 | 融合前每路召回数 |
| `SMA_CHUNK_SIZE` | 320 | 切块大小（中文字符） |
| `SMA_API_ADMIN_TOKEN` | 空 | 留空则重建索引与工具列表接口返回 401 |

切换真实服务：

```bash
SMA_LLM_PROVIDER=deepseek SMA_LLM_API_KEY=sk-xxx \
SMA_EMBEDDING_PROVIDER=deepseek_compatible SMA_EMBEDDING_API_KEY=sk-xxx \
SMA_EMBEDDING_DIM=1536 \
uv run sma index --full     # 换 embedding 必须全量重建
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 对话。身份走 `X-User-Id` 头（由认证层注入） |
| `GET` | `/api/v1/knowledge/status` | 知识库状态与文档版本 |
| `POST` | `/api/v1/knowledge/reindex` | 重建索引（**需 `X-Admin-Token`**） |
| `GET` | `/api/v1/tools` | 工具及其权限属性（需管理员，用于审计能力边界） |
| `GET` | `/health` | 健康检查 |

```bash
# 知识问答
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"7 天无理由退货从什么时候开始算","include_trace":true}' | python3 -m json.tool

# 订单查询（需要身份）
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' -H 'X-User-Id: U001' \
  -d '{"question":"查一下订单 SO20261001001 的状态"}' | python3 -m json.tool

# 越权：U001 查 U002 的订单 → 403 语义的拒绝
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' -H 'X-User-Id: U001' \
  -d '{"question":"查一下订单 SO20261005004 的状态"}' | python3 -m json.tool
```

## 测试与评测

```bash
uv run pytest                   # 174 个离线测试
uv run pytest --cov=sma         # 覆盖率
uv run pytest -m integration    # 需真实密钥，会产生费用
uv run ruff check . && uv run ruff format --check .
uv run sma eval                 # 评测集
uv run python scripts/eval_sweep.py   # 多 K 值扫描
```

**实测结果（2026-10-16，macOS arm64，Python 3.11.16）**：
174 passed / 4 deselected，覆盖率 90%，ruff 零告警。
检索指标与行为用例结果见 [EVALUATION](docs/EVALUATION.md)（12/12 行为用例通过）。

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 分层架构与调用链 |
| [TECH_SOLUTION](docs/TECH_SOLUTION.md) | 技术方案（需求 → 方案 → 取舍 → 风险） |
| [RAG_PIPELINE](docs/RAG_PIPELINE.md) | RAG 全链路 |
| [CHUNKING_AND_EMBEDDING](docs/CHUNKING_AND_EMBEDDING.md) | 切块策略与向量化 |
| [RETRIEVAL](docs/RETRIEVAL.md) | 三种检索器与 RRF 融合 |
| [RERANKING](docs/RERANKING.md) | 重排的收益与代价（含实测数据） |
| [AGENT_WORKFLOW](docs/AGENT_WORKFLOW.md) | LangGraph 图设计 |
| [TOOLS_AND_BUSINESS_RULES](docs/TOOLS_AND_BUSINESS_RULES.md) | 工具契约与业务规则 |
| [SECURITY](docs/SECURITY.md) | 威胁模型与防护 |
| [EVALUATION](docs/EVALUATION.md) | 评测方法与真实数据 |
| [DEVELOPMENT_GUIDE](docs/DEVELOPMENT_GUIDE.md) | 开发环境与命令 |
| [TECH_DECISIONS](docs/TECH_DECISIONS.md) | 选型与取舍 |
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | 故障排查 |
| [RELEASE](docs/RELEASE.md) | 发布与回滚 |
| [ROADMAP](docs/ROADMAP.md) | 进度与已知缺陷 |
| [interview/](docs/interview/) | 面试题库 |

## 已知边界（如实声明）

1. **默认 embedding 是哈希向量，不是语义向量**。`HashEmbedding` 做的是词面相似，
   同义改写（"退差价" vs "价格保护"）主要靠 BM25 分词和混合检索兜。
   生产必须换成真实 embedding 模型，换完要全量重建索引。
2. **Reranker 是词面覆盖率重排，不是 cross-encoder**。它能提精确率，但对同义改写无能为力。
3. **没有登录体系**。身份来自 `X-User-Id` 头，假设由网关/认证层注入。
   项目做到的是"身份只能来自认证层，聊天中自述的身份永远无效"这个边界。
4. **业务数据是模拟的**，不连接任何真实客户、订单或支付系统。退款工具永远不会真正执行。
5. **没有真实模型下的质量评测**。作者没有可用的 DeepSeek 密钥，
   集成测试代码写好了但实际没跑过——文档里不会出现任何真实模型下的准确率数字。

## License

仅用于学习与面试展示。
