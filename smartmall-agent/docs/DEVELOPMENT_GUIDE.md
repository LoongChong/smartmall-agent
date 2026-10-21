# 开发指南

## 1. 环境准备

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # 安装 uv
cd smartmall-agent
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env
```

Python 要求 `>=3.11,<3.13`。

**3.11 是硬下限**，不是偏好：代码用了 `asyncio.timeout`（3.11 新增）和
`StrEnum`（3.11 新增）。3.10 上会直接 ImportError。

上限 3.13 是因为 chromadb 的依赖链在 3.13 上还不稳定，没有实测通过。

> 实测环境（2026-10-16）：macOS-27.0.1-arm64，Python 3.11.16，uv 0.12.10。

## 2. 第一次跑通（零密钥）

```bash
uv run pytest                                            # 174 个离线测试
uv run sma index                                         # 构建知识库 → 6 篇 / 28 块
uv run sma ask "7 天无理由退货从什么时候开始算"            # 知识问答
uv run sma ask "查一下订单 SO20261001001 的状态" --user U001   # 工具调用
uv run sma eval                                          # 评测集
uv run sma serve                                         # http://127.0.0.1:8000/docs
```

默认配置是 `SMA_LLM_PROVIDER=mock` + `SMA_EMBEDDING_PROVIDER=hash`，
**不需要任何 API Key**。这是刻意的设计：没有密钥就跑不了任何测试的项目，
CI 也就形同虚设。

## 3. 目录结构

```
src/sma/
├── config.py              # Settings（env 前缀 SMA_，密钥用 SecretStr）
├── errors.py              # 错误分类：Retryable / Permanent / User
├── models.py              # 领域模型（Chunk / Order / ChatAnswer / RetrievalTrace ...）
├── logging_setup.py       # 结构化 JSON 日志 + contextvars 透传 trace_id（走 stderr）
├── cli.py                 # index / ask / retrieve / eval / serve
│
├── knowledge/             # 离线构建
│   ├── loader.py          #   frontmatter 解析、元数据推断、doc_id 去重
│   ├── chunker.py         #   标题优先切块 + 超长按句二切
│   └── indexer.py         #   增量/全量构建，IndexReport
│
├── rag/                   # 检索
│   ├── embedding.py       #   Embedder 抽象 + HashEmbedding / ApiEmbedding
│   ├── vector_store.py    #   Chroma 封装
│   ├── bm25.py            #   jieba 分词 + Okapi BM25
│   └── retriever.py       #   四种模式 + RRF 融合 + LexicalReranker + build_context
│
├── agent/                 # Agent
│   ├── prompts.py         #   系统提示词（集中管理，便于审计）
│   ├── guards.py          #   确定性防护：注入检测 + 相关性门槛
│   ├── validator.py       #   引用真实性 + 无依据数字检测
│   ├── workflow.py        #   LangGraph 图 + AgentState + INTENT_ROUTES
│   └── service.py         #   ChatService（图节点实现 + 编排）
│
├── tools/                 # 业务能力
│   ├── base.py            #   BaseTool 五段式 + ToolContext + ToolRegistry
│   ├── business_rules.py  #   规则引擎（退货资格、订单归属）
│   ├── business_tools.py  #   6 个工具
│   └── mock_data.py       #   模拟数据（固定时间基准 NOW）
│
├── llm/                   # 模型客户端
│   ├── base.py            #   LLMClient 抽象 + complete / complete_json
│   ├── mock.py            #   确定性 mock（离线可测的关键）
│   ├── deepseek.py        #   OpenAI 兼容适配器
│   └── factory.py         #   按配置装配
│
├── persistence/
│   └── session_store.py   # SQLite 会话（绑定 user_id，跨用户 403）
│
├── api/                   # HTTP
│   ├── app.py             #   应用装配 + trace 中间件 + 统一错误处理
│   ├── routes.py          #   路由 + require_admin
│   └── schemas.py         #   请求/响应模型
│
└── eval/                  # 评测
    ├── dataset.py         #   24 条人工标注用例
    ├── metrics.py         #   Recall@K / MRR / Precision@K / 分位数
    └── runner.py          #   检索对比 + 行为检查
```

依赖方向：`api → agent → {rag, tools} → {knowledge, llm, persistence} → {config, models, errors}`。
**不允许反向依赖**。`rag` 和 `tools` 互不依赖（它们是 Agent 的两种并列能力）。

## 4. 常用命令

```bash
# 知识库
uv run sma index                  # 增量（版本未变则跳过）
uv run sma index --full           # 全量重建（换 embedding 模型后必须）

# 对话
uv run sma ask "问题"
uv run sma ask "问题" --user U001  # 模拟已认证用户

# 只看检索（排查检索问题的首选工具）
uv run sma retrieve "退货运费" --mode hybrid --top-k 5
uv run sma retrieve "退货运费" --mode bm25        # 对比单路

# 测试
uv run pytest                     # 离线，174 个
uv run pytest -v -k retrieval     # 按名字筛
uv run pytest --cov=sma --cov-report=term-missing
uv run pytest -m integration      # 需真实密钥，会产生费用

# 静态检查
uv run ruff check .
uv run ruff format .
uv run ruff check . --fix

# 评测
uv run sma eval > /tmp/eval.json          # 日志走 stderr，不会污染这个重定向
uv run python scripts/eval_sweep.py       # K=1..4 扫描，落盘到 docs/benchmarks/

# 服务
uv run sma serve --host 0.0.0.0 --port 8000
```

## 5. 测试组织

| 文件 | 行数 | 覆盖什么 |
|------|-----:|----------|
| `conftest.py` | 105 | fixture：临时 Chroma / SQLite、mock service、工具上下文 |
| `test_knowledge.py` | 240 | frontmatter 解析、切块、索引幂等、增量更新、孤儿块清理 |
| `test_retrieval.py` | 213 | BM25 打分、RRF 融合、重排、build_context、trace |
| `test_agent.py` | 281 | 意图路由、注入短路、相关性门槛、引用校验、降级 |
| `test_tools.py` | 320 | 五段式各段、越权、高风险拦截、幂等、业务规则 |
| `test_api.py` | 243 | 路由、管理员鉴权、会话隔离、错误响应 |
| `test_http_clients.py` | 255 | httpx MockTransport：重试、超时、错误分类、JSON 解析 |
| `test_eval.py` | 121 | 指标算法、评测执行器 |
| `test_integration_real.py` | 99 | **需真实密钥，默认跳过**（`-m integration`） |

配置：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"                        # 不用给每个 async 测试加装饰器
addopts = "-q --strict-markers -m 'not integration'"
markers = ["integration: 需要真实 LLM / Embedding 服务的集成测试，默认跳过"]
```

`--strict-markers` 的作用：打错 marker 名字会直接报错，而不是静默创建一个新 marker
（那样 `-m integration` 就筛不到，集成测试会在 CI 里被意外执行）。

默认排除 integration 的理由：这些测试会**真的调 API、真的花钱**。
默认跑全量会让 `pytest` 在没有密钥时失败、有密钥时产生费用。

### 实测结果（2026-10-16）

```
174 passed, 4 deselected          覆盖率 90%
ruff check .                      All checks passed
```

4 个 deselected 是集成测试。它们**从未被实际执行过**——
作者没有可用的 DeepSeek 密钥。这一点在 [EVALUATION](EVALUATION.md) 和 README 都如实标注。

### 写测试的约定

1. **先写能复现缺陷的测试，再修**。项目里两个真实缺陷
   （相关性门槛缺失、注入未拦截）都是评测用例先失败，再改代码。
2. **不为了让测试过而改期望值**。如果断言失败，先判断是测试错了还是代码错了。
3. **外部调用一律用 `httpx.MockTransport`**，不要 mock 掉整个客户端类——
   那样测不到真实的请求构造逻辑。
   > 这条有实际价值：`test_http_clients.py` 曾暴露出 DeepSeek 适配器的真实缺陷——
   > 注入自定义 client 时 `base_url` 和 Authorization header 被忽略。
   > 如果 mock 掉整个类，这个 bug 就永远测不出来。
4. **时间相关的断言用固定基准**（`mock_data.NOW`），不用 `datetime.now()`。
5. **排序断言要有 tie-break**，否则分数相同时结果不确定。

## 6. 代码规范

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "ASYNC", "S", "C4", "RET", "SIM"]
ignore = ["S101", "B008"]
```

- `line-length = 120`：中文注释密度高，88 会导致大量无意义的折行。
- `S` 是 flake8-bandit（安全）。`S101`（assert）忽略是因为测试里必须用；
  `B008`（函数默认值里调用函数）忽略是因为 FastAPI 的 `Depends()` 就是这个写法。
- `tests/*` 额外忽略 `S` 和 `E501`。

### noqa 的使用约定

**每个 `# noqa` 必须在同一处写明理由。** 项目里现有的三处：

```python
timeout: float = (10.0,)  # noqa: ASYNC109
# ASYNC109 建议用 asyncio.timeout 替代 timeout 参数；这里 timeout 是工具的业务配置项
# （由 settings.tool_timeout_seconds 注入），内部确实是用 asyncio.timeout 实现的。
```

```python
except Exception as exc:  # noqa: BLE001 - 工具异常不得击穿到用户
```

```python
except Exception:  # noqa: BLE001 - 单篇失败不应中断整体构建
```

没有理由的 noqa 等于把规则关掉。

### 注释写什么

照抄周边代码的密度。重点注释**为什么**，不是**是什么**：

```python
# ❌ 取出问题的分词
question_terms = set(tokenize(question))

# ✅ 用单块最大覆盖率而不是全部块的并集——因为答案通常在一个块里，
#    用并集会让"每块各命中一个词"的离散噪声也通过门槛。
```

决策性的注释写在 docstring 里（`relevance_guard` 的阈值由来、BM25 的 `k1`/`b` 含义）。
架构级的决策写进 [TECH_DECISIONS](TECH_DECISIONS.md)。

## 7. 常见开发任务

| 任务 | 入口 | 配套 |
|------|------|------|
| 加知识文档 | 新建 `data/knowledge/*.md`（带 frontmatter） | `sma index`；考虑加评测用例 |
| 加业务工具 | 继承 `BaseTool` → `build_registry` | `INTENT_TOOL_MAP`、`TOOL_REQUIRED_SLOTS`、评测用例 |
| 加意图 | `Intent` 枚举 → `INTENT_SYSTEM_PROMPT` → `INTENT_ROUTES` | 评测用例（防抢分类） |
| 换 embedding | `Embedder` 实现 → `llm/factory.py` | **`sma index --full`** |
| 换向量库 | 替换 `ChromaVectorStore` | 接口：upsert/query/delete_doc/all_chunks/count/doc_versions/reset |
| 换重排器 | `Reranker` 实现 → 注入 `HybridRetriever` | 重跑 `eval_sweep.py` 对比 |
| 改切块参数 | `SMA_CHUNK_SIZE` / `SMA_CHUNK_OVERLAP` | **chunk_id 会全变**，要重建索引 |
| 调相关性门槛 | `relevance_guard(min_coverage=...)` | 必须跑评测集回归 |

详细步骤分别见 [TOOLS_AND_BUSINESS_RULES](TOOLS_AND_BUSINESS_RULES.md) 第 6 节、
[AGENT_WORKFLOW](AGENT_WORKFLOW.md) 第 8 节、
[CHUNKING_AND_EMBEDDING](CHUNKING_AND_EMBEDDING.md) 第 5.3 节。

## 8. 调试技巧

### 看检索发生了什么

```bash
uv run sma retrieve "退货运费谁承担" --mode hybrid --top-k 5
```

输出每条候选的 chunk_id、分数、来源（`vector` / `bm25` / `vector+bm25`）、内容前 100 字。

HTTP 侧用 `include_trace`：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"退货运费谁承担","include_trace":true}' | python3 -m json.tool
```

`retrieval_trace` 里能看到 `vector_candidates` / `bm25_candidates` / `fused_order` /
`reranked_order` / `final_chunk_ids` / `context_chars`——
一眼分辨是"没召回到"、"召回了没进上下文"还是"进了上下文模型没用"。

### 关掉 JSON 日志看人话

```bash
SMA_LOG_JSON=false SMA_LOG_LEVEL=DEBUG uv run sma ask "问题"
```

### 用独立的数据目录，不污染主库

```bash
SMA_DB_PATH=/tmp/t.db SMA_CHROMA_DIR=/tmp/t-chroma uv run sma index
```

CI 里就是这么做的。

### 对比两种检索模式

```bash
for m in vector bm25 hybrid hybrid_rerank; do
  echo "--- $m ---"
  SMA_RETRIEVAL_MODE=$m uv run sma retrieve "退差价" --mode $m --top-k 3
done
```

同义改写问不出来时，这个对比能立刻判断是向量路的问题（hash embedding 无语义能力）
还是语料本身没有相关内容。

更系统的排查流程见 [TROUBLESHOOTING](TROUBLESHOOTING.md)。

## 9. 提交前检查

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov=sma
uv run sma index && uv run sma eval          # 端到端 + 评测不回归
grep -rniE '(sk-[a-zA-Z0-9]{10,}|api[_-]?key\s*=\s*["'"'"'][^"'"'"']{8,})' src/ tests/ scripts/
```

最后那条是硬编码密钥扫描，无输出即通过。

CI（`.github/workflows/ci.yml`）跑的是同一套：ruff → pytest（带覆盖率）→
端到端 smoke（`sma index` + `sma ask`）→ 评测集回归 → Docker build + 健康检查。

注意 CI 里 `ruff format --check` 后面带了 `|| true`——
**格式化只提示不阻塞**，而 `ruff check`（lint）是硬门禁。
这是刻意区分的：格式问题不影响正确性，不值得卡住流水线。
