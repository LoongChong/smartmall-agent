# 故障排查

每条都给出**症状 → 怎么确认 → 怎么修**，不是泛泛的"检查配置"。

## 0. 先做这三步

```bash
# ① 服务与索引状态
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/v1/knowledge/status | python3 -m json.tool

# ② 带 trace 复现一次
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"你的问题","include_trace":true}' | python3 -m json.tool

# ③ 看人话日志
SMA_LOG_JSON=false SMA_LOG_LEVEL=DEBUG uv run sma ask "你的问题"
```

`health` 里的 `knowledge_ready` 和 `chunk_count` 能排掉一半的问题。

## 1. 安装与启动

### `uv pip install -e .` 报 `Readme file does not exist`

`pyproject.toml` 声明了 `readme = "README.md"` 但文件不在。
确认在仓库根目录执行，且 `README.md` 存在。

### `ImportError: cannot import name 'timeout' from 'asyncio'`

Python < 3.11。`asyncio.timeout` 和 `StrEnum` 都是 3.11 新增的。

```bash
uv python install 3.11 && uv venv --python 3.11 && uv pip install -e ".[dev]"
python -c "import sys; print(sys.version)"
```

### `KnowledgeBaseError: 未安装 chromadb`

```bash
uv pip install -e ".[dev]"
```

注意要带 `[dev]`——虽然 chromadb 在主依赖里，但这个报错通常意味着装到了错误的 venv。
确认 `which python` 指向项目的 `.venv`。

### 服务启动了但 `knowledge_ready: false`

应用启动时会自动跑一次增量索引（`app.py` 的 lifespan）。
失败不阻止启动，但会记 `startup_index_failed`。查日志：

```bash
uv run sma serve 2>&1 | grep -i index
```

常见原因：`SMA_KNOWLEDGE_DIR` 指向了不存在的目录（容器里路径不同）。

## 2. 知识库与索引

### 所有问题都答"知识库中没有找到相关信息"

**第一嫌疑是索引为空。**

```bash
curl -s http://127.0.0.1:8000/api/v1/knowledge/status | python3 -m json.tool
# chunk_count: 0  →  没索引
```

```bash
uv run sma index          # 期望输出 chunks_written: 28（6 篇示例文档）
```

如果 `chunk_count > 0` 但依然全部拒答，看第 3 节的相关性门槛。

> 为什么向量检索会抛 `KnowledgeBaseEmptyError`（503）而不是返回空：
> 索引为空是**配置问题**，不是"没有答案"。静默返回空会让部署时忘了跑
> `sma index` 这件事被完全掩盖。

### `IndexingError: 知识库目录不存在` / `目录下没有 .md 文档`

同样是故意抛错而不是返回空列表。检查 `SMA_KNOWLEDGE_DIR`：

```bash
echo $SMA_KNOWLEDGE_DIR; ls -la data/knowledge
```

容器里默认是 `/app/data/knowledge`（见 Dockerfile 的 ENV）。

### `IndexingError: 重复的 doc_id`

两篇文档的 frontmatter 里 `doc_id` 相同，或者两个文件名相同（doc_id 缺失时用文件名）。
chunk_id 由 doc_id 派生，重复会互相覆盖，所以必须拒绝。

### 改了文档内容但检索结果没变

增量模式按**内容哈希**判断是否需要重建。如果确实改了内容：

```bash
uv run sma index          # 看 docs_indexed 是否 > 0
```

`docs_skipped=6, docs_indexed=0` 说明哈希没变——确认改的是 `SMA_KNOWLEDGE_DIR`
指向的那个目录里的文件（容器挂载和本地路径容易搞混）。

实在不确定就全量：

```bash
uv run sma index --full
```

### 检索到了已经删掉的内容

两种情况：

1. **文档文件被删了但索引没清**：跑一次 `sma index`（增量模式会对比 doc_id 差集并清理）。
2. **文档变短后残留孤儿块**：这个已经在 `indexer.py` 里修了
   （内容变化时先 `delete_doc` 再写）。如果仍然出现，`sma index --full`。

### 换了 embedding 模型后检索结果全是乱的

**必须全量重建**：

```bash
uv run sma index --full
```

原因：增量模式只比对内容哈希，**感知不到 embedding 模型变了**。
不同模型的向量空间完全不可比；维度不同会直接报 Chroma 的维度错误，
维度恰好相同的话会算出完全无意义的相似度——而且这种错误是静默的，
索引看起来构建成功了。

同时确认 `SMA_EMBEDDING_DIM` 改成了新模型的实际维度。

## 3. 检索与回答质量

### 同义改写问不出来（"退差价"查不到"价格保护"）

这是**已知边界**，不是 bug。确认方式：

```bash
uv run sma retrieve "退差价" --mode bm25 --top-k 3
uv run sma retrieve "退差价" --mode vector --top-k 3
```

如果 BM25 能召回而 vector 不能 → 默认的 `HashEmbedding` 是**词面哈希，不是语义向量**，
对没有共享词的同义表达无能为力。

三个选项：

1. 用默认的 `hybrid`（BM25 兜住一部分，jieba 会把"退差价"切出"差价"）；
2. 在知识库文档里补上同义表述（最直接有效）；
3. 换真实 embedding：

```bash
SMA_EMBEDDING_PROVIDER=deepseek_compatible SMA_EMBEDDING_API_KEY=sk-xxx \
SMA_EMBEDDING_DIM=1536 uv run sma index --full
```

### 明明知识库里有，却被拒答

看日志里的相关性门槛事件：

```bash
SMA_LOG_JSON=false uv run sma ask "你的问题" 2>&1 | grep -E "relevance_guard_blocked|no_answer_by_guard"
# {"event": "relevance_guard_blocked", "coverage": 0.25, "threshold": 0.34, ...}
```

`coverage` 低于 0.34 就会在**调模型之前**直接拒答。

`relevance_guard` 的算法是：问题实词与**单个块**的词面交集比例。
所以"问题用词和文档用词完全不同"时会误杀。

怎么处理：

- **优先改文档**，让文档包含用户实际会用的说法。这是治本的。
- 临时放宽阈值（`agent/guards.py` 的 `min_coverage` 默认值），
  但**必须跑 `uv run sma eval` 回归**——放宽它会让无关问题重新开始得到编造的答案，
  这个门槛本来就是为了修那个缺陷才加的。

### 回答里有知识库里没有的数字

```bash
SMA_LOG_JSON=false uv run sma ask "问题" 2>&1 | grep unsupported_numbers
```

`find_unsupported_numbers` 只告警不拦截（可能是合理的单位换算）。
但**频繁出现**说明提示词约束不够，或者上下文被截断导致模型靠猜。

先看 trace 的 `context_chars` 是否接近 `SMA_CONTEXT_MAX_CHARS`（默认 4000）——
如果顶到上限，说明有块被丢了。

### `dropped_citations` 非空

```json
{"status": "answered", "citations": [...], "dropped_citations": ["fake_doc#9"]}
```

**这是防护生效，不是故障。** 模型编了一个不存在的 chunk_id，被引用校验剔除了。

如果剔除后一条真引用都不剩，状态会降级成 `no_answer` 并换成拒答话术——
因为"声称有依据但一条真引用都没有"比"说不知道"更危险。

频繁出现要加强 `ANSWER_SYSTEM_PROMPT` 里的引用约束，或检查上下文里的 chunk_id 格式
是否被破坏（`build_context` 的格式是 `[chunk_id] 《标题》`）。

### 召回正确但回答跑偏

看 trace：

```
fused_order:      ["a#1", "a#2", "b#0", "c#3"]
final_chunk_ids:  ["a#1", "a#2"]            ← 只有 2 个，后两个被丢了
context_chars:    3980                       ← 顶到 4000 上限
```

上下文被截断。`build_context` 是**整块纳入或整块丢弃**（不截半块，否则 chunk_id
和内容会错配）。调大 `SMA_CONTEXT_MAX_CHARS` 或调小 `SMA_CHUNK_SIZE`。

### 无关问题也给出了答案

```bash
uv run sma ask "你们公司 CEO 是谁"     # 期望：拒答
```

如果它给出了答案，说明相关性门槛被调低或被绕过。跑评测集确认：

```bash
uv run sma eval | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['behavior']['pass_rate'])"
```

> 这正是评测用例 `b01` 当初暴露的真实缺陷。修复是加 `relevance_guard`，
> 不是改评测期望。见 [EVALUATION](EVALUATION.md)。

## 4. 工具与权限

### 查订单返回"无法确认您的身份，请先登录后再查询订单"

没有传身份。

```bash
curl -X POST .../chat -H 'X-User-Id: U001' -d '{"question":"查订单 SO20261001001"}'
uv run sma ask "查订单 SO20261001001" --user U001
```

`order_query` 等工具 `requires_auth = True`。

### 查自己的订单也返回"未找到该订单，或该订单不属于当前登录账号"

这条话术**同时覆盖两种情况**（订单不存在 / 订单不属于你）——
这是故意的，防止攻击者遍历订单号枚举出哪些是真实的。

所以要区分，得看日志：

```bash
SMA_LOG_JSON=false uv run sma ask "查订单 SOxxx" --user U001 2>&1 | grep -E "tool_forbidden|order_not_found"
```

`tool_forbidden` → 订单存在但不属于你；`order_not_found` → 订单不存在。

模拟数据里的订单归属见 `src/sma/tools/mock_data.py`（`U001` 有 `SO20261001001` 等）。

### 工具没被调用，走了知识库分支

意图识别没认出来。看日志：

```bash
SMA_LOG_JSON=false uv run sma ask "问题" --user U001 2>&1 | grep -E "intent_routed|intent_fallback"
```

- `intent_fallback` → **模型调用失败了**，降级为知识库检索（这是最安全的兜底：
  知识库分支不触发任何工具和写操作）。检查 `SMA_LLM_API_KEY`。
- `intent_routed` 显示的意图不对 → 问法太模糊，或者 mock 模型的规则覆盖不到。
  mock 模型是关键词匹配的，`SMA_LLM_PROVIDER=mock` 下不要期待复杂问法能被正确分类。

### 一直返回"需要您提供订单号"

槽位抽取失败。`TOOL_REQUIRED_SLOTS` 要求 `order_query` 必须有 `order_id`，
缺失时走澄清（不让工具白跑）。

订单号格式校验在 `OrderQueryArgs`：长度 4–40，只允许字母数字和连字符。
确认问题里的订单号格式正确（如 `SO20261001001`）。

### 退款请求总是转人工

**设计如此。** `RefundTool` 的 `is_high_risk=True`，
而且退款意图被路由到 `escalate` 根本到不了工具层。
即使到了，`BaseTool.execute` 第 3 段也会拦住（`approved_high_risk` 没有任何代码路径会设为 True）。

见 [TOOLS_AND_BUSINESS_RULES](TOOLS_AND_BUSINESS_RULES.md) 第 2 节。

### 创建售后报"请勿重复提交"

幂等键命中：`sha256(user_id | order_id | reason)`。同一用户、同一订单、同一原因只能提交一次。
换一个原因（不同的售后诉求）可以创建新工单。

**注意幂等存储是进程内字典**，重启后清空，多实例不共享。测试里用 `reset_tickets()` 清理。

## 5. API 与服务

### 管理接口返回 401 "管理接口未启用"

```bash
SMA_API_ADMIN_TOKEN=your-token uv run sma serve
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/reindex \
  -H 'X-Admin-Token: your-token' -H 'Content-Type: application/json' \
  -d '{"mode":"incremental"}'
```

**令牌未配置时直接拒绝，而不是放开。** 这是"默认安全"——
忘了配置的后果是接口不可用（能被立刻发现），而不是接口对所有人开放。

### 想关掉管理员鉴权

```bash
SMA_REINDEX_REQUIRES_ADMIN=false
```

**不要在生产这么做。** 全量重建是 DoS 放大器（一个请求放大成几百次 embedding 调用）。

### 503 `knowledge_base_empty`

索引为空。见第 2 节。这个错误被**单独放行**不转成通用错误，就是为了让它可见。

### 跨会话读取返回 403

```python
if row["user_id"] != user_id:
    raise ForbiddenError("该会话不属于当前用户")
```

会话绑定 `user_id`。用 `X-User-Id: U001` 创建的 session_id，
换成 `U002` 去读会被拒。日志里有 `session_owner_mismatch`。

### 想追踪一个请求的全链路

每个响应都带 `X-Trace-Id` 头和 body 里的 `trace_id`。
也可以自己传入：

```bash
curl -H 'X-Trace-Id: my-debug-001' ...
grep my-debug-001 logs.jsonl
```

`trace_id` 通过 contextvars 透传到所有日志事件。

## 6. 模型调用

### `config_error: LLM provider=deepseek 需要配置 SMA_LLM_API_KEY`

```bash
SMA_LLM_PROVIDER=mock          # 不用密钥
# 或
SMA_LLM_API_KEY=sk-xxx
```

不静默降级到 mock，因为那会让人以为在用真实模型。

### `llm_timeout` / `llm_rate_limit`

这两个是 `RetryableError`，客户端内部已经重试了 `SMA_LLM_MAX_RETRIES` 次（默认 2，指数退避）。
仍然失败说明服务端持续不可用。

```bash
SMA_LLM_TIMEOUT_SECONDS=90 SMA_LLM_MAX_RETRIES=4
```

### `llm_output_format_error`

模型没返回合法 JSON。`complete_json` 会尝试提取被 markdown 代码块包裹的 JSON，
但模型输出严重偏离格式时仍会失败。

降低温度：`SMA_LLM_TEMPERATURE=0`。

### 模型挂了但订单查询还能用

**设计如此。** `do_answer_from_tool` 里有 fallback：

```python
except (SMAError, LLMError):
    text = summary      # 直接用工具摘要
```

工具返回的数据是确定性的，不依赖模型。模型只是把它说得更口语化。
日志里有 `tool_answer_fallback`。

## 7. 测试与评测

### `uv run sma eval > report.json` 得到的不是合法 JSON

应该已经修好了：**日志走 stderr**，stdout 只有 JSON 输出。
如果仍有问题，确认没有第三方库往 stdout 写东西
（jieba 的词典加载进度曾经是罪魁祸首，已用 `jieba.setLogLevel("WARNING")` 压住）。

```bash
uv run sma eval 2>/dev/null | python3 -m json.tool    # 丢弃 stderr 验证
```

### 评测指标和文档里的不一致

检查环境是否一致：

```bash
uv run sma eval | python3 -c "import json,sys; print(json.load(sys.stdin)['environment'])"
```

`embedding_provider`、`chunk_size`、`top_k` 任意一项不同，数字就不可比。
文档里的数据全部是 `hash` + `chunk_size=320` + `rrf_k=60` 下跑出来的。

### 行为用例失败

```bash
uv run sma eval | python3 -c "
import json,sys
d = json.load(sys.stdin)
for x in d['behavior']['details']:
    if not x['pass']: print(x['case_id'], x['question'], x['expected_status'], '->', x['actual_status'])
"
```

**失败了先判断是代码错还是期望错。** 项目历史上两次行为用例失败
（`b01` 无关问题被编答案、`b10` 注入未拦截）都是**代码的真实缺陷**，
修复方式是加 `agent/guards.py`，不是改评测期望。

### 测试里 Chroma 报 collection 冲突

`conftest.py` 给每个测试用临时目录。如果自己写测试，确保用了 fixture
而不是共享默认的 `data/chroma`。

### `pytest -m integration` 全部失败

需要真实密钥：

```bash
SMA_LLM_PROVIDER=deepseek SMA_LLM_API_KEY=sk-xxx \
SMA_EMBEDDING_PROVIDER=deepseek_compatible SMA_EMBEDDING_API_KEY=sk-xxx \
SMA_EMBEDDING_DIM=1536 uv run pytest -m integration
```

**会产生真实费用。** 这些测试在本项目里**从未被实际执行过**（没有可用密钥）。

## 8. Docker

### 容器起来但 `/health` 不通

```bash
docker logs <container>
docker exec -it <container> ls -la /data /app/data/knowledge
```

常见原因：`/data` 权限。镜像里 `chown sma:sma /data`，但如果挂载了主机目录，
主机目录的 owner 要能被 uid 10001 写入。

### 容器里 `knowledge_ready: false`

镜像里知识库在 `/app/data/knowledge`（Dockerfile 的 `SMA_KNOWLEDGE_DIR`）。
如果覆盖了这个环境变量指向别处，而那里没有文件，索引就是空的。

### 数据重启后丢失

```yaml
volumes:
  - sma-data:/data
```

`SMA_DB_PATH=/data/sma.db`、`SMA_CHROMA_DIR=/data/chroma` 都在卷里。
没挂卷的话容器删除后数据就没了。

## 9. 日志事件速查

| 事件 | 含义 | 要不要紧 |
|------|------|---------|
| `docs_loaded` | 文档加载完成 | 正常 |
| `doc_category_inferred` | 元数据缺失，按文件名推断了分类 | 建议补 frontmatter |
| `index_done` | 索引完成，带完整报告 | 正常 |
| `index_doc_failed` | 单篇文档索引失败 | **要查**，整体仍会继续 |
| `startup_index_failed` | 启动时自动索引失败 | **要查**，服务仍会启动 |
| `vector_store_reset` | 全量重建，collection 被删除重建 | 确认是预期操作 |
| `bm25_built` | BM25 索引重建 | 正常 |
| `retrieval_done` | 检索完成，带各路候选数 | 正常 |
| `intent_routed` | 意图路由结果 | 正常 |
| `intent_fallback` | **模型调用失败**，降级为知识库检索 | **要查** |
| `intent_unknown` | 模型输出了枚举外的意图 | 关注频率 |
| `injection_detected` | 注入模式命中，已短路 | 防护生效 |
| `identity_spoof_attempt` | 身份冒充话术，仅告警 | 关注频率 |
| `relevance_guard_blocked` | 相关性不足，拒答 | 可能误杀，看 coverage |
| `no_answer_by_guard` | 同上，在回答节点 | 同上 |
| `tool_invalid_args` | 工具参数校验失败 | 槽位抽取有问题 |
| `tool_unauthorized` | 工具需要身份但没有 | 正常拦截 |
| `tool_forbidden` | 越权访问被拦 | 防护生效，关注频率 |
| `tool_high_risk_blocked` | 高风险操作被拦 | 防护生效 |
| `tool_timeout` | 工具执行超时 | **要查** |
| `tool_unexpected_error` | 工具抛了未预期异常 | **要查**，带堆栈 |
| `tool_answer_fallback` | 模型挂了，直接用工具摘要 | **要查**模型 |
| `citation_dropped` | 模型编了引用，已剔除 | 防护生效，关注频率 |
| `answer_without_citation` | 声称有答案但无真引用，已降级拒答 | 关注频率 |
| `unsupported_numbers` | 回答里有上下文中不存在的数字 | 关注频率 |
| `session_owner_mismatch` | 跨用户访问会话 | 防护生效，**可能是攻击** |
| `chat_done` | 对话完成，带状态与耗时 | 正常 |
| `chat_crashed` | 未预期异常，已兜底 | **要查**，带堆栈 |

## 10. 还是定位不了

把这些一起带上：

1. `curl -s http://127.0.0.1:8000/health`
2. `curl -s http://127.0.0.1:8000/api/v1/knowledge/status`
3. 带 `include_trace: true` 的完整响应（含 `trace_id`）
4. `SMA_LOG_JSON=false SMA_LOG_LEVEL=DEBUG` 下复现的日志
5. `uv run sma eval` 的 `environment` 和 `behavior.pass_rate`
6. `python -c "import sys; print(sys.version)"` 和 `uv --version`
