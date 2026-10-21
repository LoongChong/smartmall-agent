# 进度与已知缺陷

## 一、已完成（v0.1.0，2026-10-16）

### 知识库

- [x] Markdown + YAML frontmatter 加载（手写极简解析，不依赖 PyYAML）
- [x] 元数据缺失时按文件名推断分类，不中断构建
- [x] `doc_id` 重复、正文为空、目录不存在 → 显式报错（不静默返回空）
- [x] 按 Markdown 标题优先切块 + 超长小节按句二次切分带重叠
- [x] 内容哈希做版本，增量更新幂等（重复执行写入量为 0）
- [x] 内容变短时先删旧块，避免孤儿块
- [x] 磁盘上删除的文档，索引里同步清理
- [x] 单篇失败不中断整体，失败明细进 `IndexReport.failures`
- [x] 6 篇原创中文电商知识文档（商品规格 ×2、售后政策、保修、支付 FAQ、通用 FAQ）

### 检索

- [x] `Embedder` 抽象 + `HashEmbedding`（离线）/ `ApiEmbedding`（OpenAI 兼容）
- [x] Chroma 封装（嵌入在外部计算，避免默认下载 ONNX 模型）
- [x] jieba 分词 + 自写 Okapi BM25（k1=1.5、b=0.75、IDF 带平滑防负值）
- [x] 电商场景停用词表（含"请问""一下""怎么"这类低区分度词）
- [x] 四种检索模式：vector / bm25 / hybrid / hybrid_rerank
- [x] RRF 融合（k=60，只用排名，跨量纲可比）
- [x] `LexicalReranker`（词面覆盖率 + 位置 + 标题 + 保留原排名影响）
- [x] 全链路 `RetrievalTrace` 留痕，`include_trace` 可取
- [x] 所有排序都有 tie-break，保证可复现

### Agent

- [x] LangGraph 条件路由图，8 节点，**无回边**
- [x] 9 种意图，路由表集中在 `INTENT_ROUTES`
- [x] 确定性注入检测（19 条模式，中英文），命中即短路
- [x] 身份冒充检测（只告警，因为自述身份本来就不生效）
- [x] 相关性门槛（覆盖率 < 0.34 直接拒答，不调模型）
- [x] 引用真实性校验，伪造 ID 剔除 + 全假则降级拒答
- [x] 无依据数字检测（告警不拦截）
- [x] 意图识别失败 → 降级知识库检索（最安全的兜底）
- [x] 工具回答失败 → 直接用工具摘要

### 业务工具

- [x] `BaseTool` 五段式：参数校验 → 鉴权 → 高风险拦截 → 超时 → 异常分类
- [x] 6 个工具：商品 / 订单 / 物流 / 退货资格 / 售后创建 / 退款（永远拦截）
- [x] 退货资格规则引擎（收集全部阻塞原因，不短路返回）
- [x] 订单归属校验唯一入口，4 个工具全部调用
- [x] 运单号查询路径独立校验归属
- [x] 资源不存在 / 无权访问 **统一话术**（防订单号枚举）
- [x] 售后创建幂等（`sha256(user|order|reason)`）+ 写前重跑规则
- [x] 固定时间基准 `NOW`，时间相关断言永远可复现
- [x] `GET /api/v1/tools` 审计能力边界（需管理员）

### 接口与运维

- [x] HTTP：`/chat`、`/knowledge/status`、`/knowledge/reindex`、`/tools`、`/health`
- [x] CLI：`index` / `ask` / `retrieve` / `eval` / `serve`
- [x] 会话 SQLite 持久化，绑定 user_id，跨用户 403，历史有上限
- [x] 结构化 JSON 日志走 stderr + contextvars 透传 trace_id
- [x] 统一错误模型（Retryable / Permanent / User），不暴露堆栈
- [x] `SecretStr` 密钥脱敏，JSON 日志 `default=repr` 二次兜底
- [x] 管理接口令牌未配置时**拒绝**而非放开
- [x] Docker 多阶段构建、非 root（uid 10001）、HEALTHCHECK
- [x] GitHub Actions：ruff → pytest → 端到端 smoke → 评测回归 → Docker build

### 测试与评测

- [x] 174 个离线测试，覆盖率 90%，ruff 零告警
- [x] 24 条人工标注评测用例（12 检索 + 12 行为）
- [x] 四模式对比 + K=1..4 扫描脚本，结果落盘 `docs/benchmarks/`
- [x] 外部调用用 `httpx.MockTransport` 测真实请求构造
- [x] 4 个集成测试（标 `integration`，默认跳过）

### 文档

- [x] 18 份：README + ARCHITECTURE + TECH_SOLUTION + RAG_PIPELINE +
      CHUNKING_AND_EMBEDDING + RETRIEVAL + RERANKING + AGENT_WORKFLOW +
      TOOLS_AND_BUSINESS_RULES + SECURITY + EVALUATION + TECH_DECISIONS +
      DEVELOPMENT_GUIDE + TROUBLESHOOTING + RELEASE + ROADMAP + AGENTS + interview/

## 二、已知缺陷

**不是"将来的优化点"，是当前实现的真实不足。** 每条都写明影响和触发条件。

| # | 缺陷 | 影响 | 触发条件 | 位置 |
|---|------|------|---------|------|
| D1 | 默认 embedding 是**哈希词面向量，不是语义向量** | 同义改写召回差（"退差价" vs "价格保护"） | 默认配置 | `rag/embedding.py` |
| D2 | 换 embedding 模型后，增量模式**感知不到** | 维度不同→报错；维度相同→**静默**算出无意义相似度 | 改了 provider 却没加 `--full` | `knowledge/indexer.py` |
| D3 | 售后工单与幂等记录是**进程内字典** | 重启丢失；多实例部署下重复提交防不住 | 多实例 / 重启 | `tools/business_tools.py:_TICKETS` |
| D4 | BM25 索引**常驻内存**，进程启动时全量重建 | 语料大了启动慢、内存占用随语料线性增长 | 文档数量上千 | `knowledge/indexer.py:build_bm25` |
| D5 | **没有速率限制** | 可被刷请求、刷模型费用 | 公网暴露且网关没限流 | — |
| D6 | **没有登录体系**，身份来自 `X-User-Id` 头 | 直接暴露公网等于无鉴权 | 网关没剥离客户端头 | `api/routes.py` |
| D7 | 日志**没有字段级 PII 脱敏** | 订单号、商品名原样记录 | 有合规要求时 | `logging_setup.py` |
| D8 | 规则引擎与知识库文档**人工同步** | 两处描述同一政策，可能不一致 | 改了规则忘了改文档 | `business_rules.py` ↔ `aftersale_return_policy.md` |
| D9 | 注入模式表**会漏报变体表达** | 换一种说法可能绕过第一道闸 | 攻击者改写话术 | `agent/guards.py` |
| D10 | 相关性门槛**会误杀同义表达** | 有答案却拒答 | 用户用词与文档完全不同 | `agent/guards.py` |
| D11 | `LexicalReranker` 在 `top_k=1` 时**净亏损** | Recall/MRR/Precision 全降 8.3%（实测） | `hybrid_rerank` + `top_k=1` | `rag/retriever.py` |
| D12 | 词面重排**损害 MRR**（实测 −4.2%） | 首位准确率下降 | 开启 `hybrid_rerank` | 同上 |
| D13 | 会话历史裁剪用 `seq <= seq-keep` | 并发写入同一会话时 seq 可能冲突（有 PRIMARY KEY 兜底会报错） | 同一会话并发请求 | `persistence/session_store.py` |
| D14 | `max_tool_calls` 配置项**当前无实际约束力** | 每轮最多调 1 个工具，这个上限是为将来预留的 | — | `agent/service.py` |
| D15 | 开发依赖只有下界没有上界（`pytest>=8.3.0` 等） | 重建环境时解析到新大版本，可能破坏测试 | 重建 venv | `pyproject.toml` |

> D15 不是假设的风险：姊妹项目 Deep Research Agent 在 2026-10-16 验收时实际踩到了——
> 重建环境解析到 pytest 9.1.1，它不再把 rootdir 隐式加入 `sys.path`，
> 导致 3 个共享 fixture 的测试模块收集失败。
> 本项目的测试之间没有跨模块导入，所以没被触发，但根因相同。
> 修法：给 dev 依赖加上界，或提交 `uv.lock`。

## 三、未验证的部分

**和缺陷分开列**，因为这些不是实现问题，是"没有条件验证"。

| # | 项 | 状态 |
|---|----|------|
| U1 | 真实 LLM（DeepSeek）下的端到端行为 | 集成测试代码已写好，**从未执行**（无可用密钥） |
| U2 | 真实 embedding 下的检索指标 | 同上。文档里**不出现**任何此类数字 |
| U3 | 真实模型下概率性防护的效果 | 提示词约束的实际效果未验证；确定性防护不受影响 |
| U4 | cross-encoder 重排的收益 | `RERANKING` 里给的是接口示例代码，**未实现未实测** |
| U5 | 生产规模语料（上百篇）下的表现 | 当前 6 篇，K≥2 时 Recall 已到 1.0（天花板效应） |
| U6 | 高并发下的稳定性 | 未做压测，没有并发指标 |

## 四、下一步（按优先级）

### P0 —— 生产化必须做

1. **换真实 embedding 并重跑全部评测**（解 D1、U2）。
   这是所有检索结论的前提。换完后 `RETRIEVAL`、`RERANKING` 两份文档的数据表必须重跑。
2. **幂等与工单落库**（解 D3）。
   SQLite 加 `aftersale_tickets` 表，幂等键建唯一索引，用 `INSERT ... ON CONFLICT DO NOTHING`
   + `rowcount` 判定（姊妹项目 DRA 的证据去重就是这个写法，可以直接参照）。
3. **网关集成文档化**（解 D6）。
   明确要求：验证 JWT → 解析 user_id → **剥离客户端自带的 `X-User-Id`** → 注入真实值。
   这一步不做，所有归属校验形同虚设。
4. **执行集成测试**（解 U1）。有密钥后跑 `pytest -m integration`，把结果写进 EVALUATION。

### P1 —— 质量与可观测

5. **扩充评测集到 100+ 条，语料扩到 50+ 篇**（解 U5）。
   当前天花板效应让大部分参数调整测不出差异。这是所有调参工作的前提。
6. **接 cross-encoder 重排并实测对比**（解 U4、D11、D12）。
   注意 `CrossEncoder.predict` 是同步的，要包 `asyncio.to_thread`；
   依赖体积会让镜像涨到 GB 量级。
7. **BM25 持久化倒排索引**（解 D4）。
   语料上千后内存重建不可接受。
8. **日志 PII 脱敏规则**（解 D7）。
   字段级白名单 + 订单号部分掩码。
9. **指标埋点**（Prometheus）：意图分布、拒答率、工具成功率、引用剔除率、各节点耗时。
   拒答率和引用剔除率是质量回归的早期信号。

### P2 —— 功能与工程

10. **政策参数单一数据源**（解 D8）。把天数、例外商品类型抽成配置，
    代码和知识库文档都从它生成。当前规模下这个抽象收益不足，所以没做。
11. **多工具编排**（让 D14 真正生效）。例如"这单能退吗，退了运费谁出"
    需要同时调 `return_eligibility_check` 和检索运费政策。
12. **会话并发安全**（解 D13）。seq 改用数据库自增或加乐观锁。
13. **扩充注入模式表 + 语义注入检测**（缓解 D9）。
    模式表永远会漏，可以加一个轻量分类器做第二道闸。
14. **相关性门槛自适应**（缓解 D10）。阈值按问题长度/类型动态调整，
    但**必须有足够大的评测集**才能验证，依赖第 5 项。
15. 流式输出（SSE）、多轮上下文改写（query rewriting）、知识库热更新推送。

## 五、不打算做的

| 项 | 为什么 |
|----|--------|
| 引入 CrewAI | 单 Agent + 条件路由，无多智能体协作需求；且它的"Agent 自选工具"与本项目的安全设计冲突 |
| 用 LangGraph 做工具内部的子流程 | 工具是确定性函数，用图编排是过度设计 |
| 应用层速率限制 | 网关的标准职责，在应用层重复实现只会和网关配置打架 |
| 自己实现向量索引（HNSW） | Chroma 够用；自研的唯一收益是"能讲"，不值得维护成本 |
| 把业务规则做成可配置的规则 DSL | 当前 4 条规则，DSL 的复杂度远超收益。超过 20 条再考虑 |
| 让模型直接指定工具名 | 会打穿能力最小化这条安全设计 |
