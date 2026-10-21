# 架构设计

## 1. 目标与非目标

**目标**

1. 让客服回答**有据可查**：每条知识性结论能回指到具体文档片段。
2. 让业务结论**由代码决定**：退货能不能退、订单能不能看，不由模型说了算。
3. 检索方案可对比、可评测、可复现，而不是"调参靠感觉"。
4. 无密钥即可完整运行与测试。

**非目标**

- 不做多智能体协作（单 Agent + 条件路由够用，见 TECH_DECISIONS）。
- 不做登录体系（身份由认证层注入，项目只守住边界）。
- 不连接真实支付/退款系统。

## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│ API 层      FastAPI 路由 / Schema / 统一错误 / trace 中间件      │ src/sma/api
├──────────────────────────────────────────────────────────────┤
│ Agent 层    ChatService + LangGraph 图 + guards + validator     │ src/sma/agent
├──────────────────────────────────────────────────────────────┤
│ 能力层      RAG（检索）        │  Tools（业务工具 + 规则引擎）    │ src/sma/rag, src/sma/tools
├──────────────────────────────────────────────────────────────┤
│ 基础设施层  知识库构建 · LLM/Embedding 客户端 · Chroma · SQLite  │ src/sma/{knowledge,llm,persistence}
└──────────────────────────────────────────────────────────────┘
```

依赖方向自上而下。`rag` 与 `tools` 互不依赖——它们是 Agent 的两种**并列能力**，
由意图路由决定走哪条。这个边界很重要：知识性问题和业务数据问题的失败模式完全不同
（前者是"检索不到"，后者是"无权访问"），混在一起会让错误处理变得混乱。

## 3. 两条主链路

### 知识问答链路

```
POST /chat
 └─ ChatService.chat
     └─ LangGraph: route
         ├─ guards.detect_injection          # 确定性短路，先于模型
         └─ LLM(intent) → Intent.KNOWLEDGE_QA
     └─ retrieve
         └─ HybridRetriever
             ├─ VectorRetriever → HashEmbedding → Chroma
             ├─ BM25Retriever  → jieba 分词 → BM25 打分
             ├─ reciprocal_rank_fusion
             └─ LexicalReranker（可选）
     └─ answer
         ├─ guards.relevance_guard           # 覆盖率不足 → 直接拒答
         ├─ build_context（带 chunk_id）
         └─ LLM(answer) → {status, answer, citations}
     └─ validate
         └─ validate_citations               # 伪造引用剔除 / 全假则降级 no_answer
```

### 业务工具链路

```
POST /chat (X-User-Id: U001)
 └─ LangGraph: route → Intent.ORDER_QUERY（slots 抽出 order_id）
     └─ tool
         ├─ 必填 slot 缺失 → CLARIFY（不让工具白跑）
         └─ BaseTool.execute
             ├─ Pydantic 参数校验
             ├─ requires_auth 检查（身份来自 ToolContext，不是聊天内容）
             ├─ is_high_risk 拦截（退款类直接转人工）
             ├─ asyncio.timeout
             └─ _run → assert_order_ownership → 业务数据
     └─ tool_answer
         └─ LLM 只负责把结构化数据说成人话；模型挂了直接用工具摘要
     └─ validate（工具分支无知识库引用，跳过）
```

## 4. 关键模块

| 模块 | 职责 | 关键设计 |
|------|------|----------|
| `knowledge/loader` | 加载 Markdown + frontmatter | 元数据缺失时推断而非报错；重复 doc_id 直接拒绝 |
| `knowledge/chunker` | 切块 | **按标题优先**切分，对齐语义边界；超长小节按句二次切分带重叠 |
| `knowledge/indexer` | 索引构建 | 内容哈希做版本，增量更新幂等；内容变短先删旧块避免孤儿 |
| `rag/embedding` | 向量化 | 抽象接口 + 两种实现；默认 hash（离线可测） |
| `rag/bm25` | 词法检索 | jieba 分词 + 标准 Okapi BM25，参数与停用词全部暴露 |
| `rag/retriever` | 检索编排 | RRF 融合（只用排名，跨量纲可比）+ 可选重排 + 全程留痕 |
| `agent/guards` | 确定性防护 | 注入模式匹配短路；相关性门槛防"无中生有" |
| `agent/validator` | 输出校验 | 引用真实性；全假引用降级为拒答；无依据数字告警 |
| `tools/base` | 工具框架 | 校验/鉴权/高风险/超时/异常五段式，统一产出 ToolCallRecord |
| `tools/business_rules` | 规则引擎 | 退货资格、订单归属——**确定性代码，不经模型** |
| `persistence/session_store` | 会话 | 绑定 user_id，跨用户读取 403；历史有上限 |

## 5. 状态与数据

- **知识库状态**：Chroma（向量 + 元数据），BM25 索引在内存中从 Chroma 重建，
  两者共用同一份语料，不会出现"向量里有、BM25 里没有"的不一致。
- **会话状态**：SQLite，`sessions` + `turns` 两张表。
- **业务数据**：`tools/mock_data.py` 中的模拟数据，固定时间基准 `NOW`，保证测试可复现。
- **售后工单**：进程内字典（`_TICKETS`），这是演示级实现，已记在 ROADMAP。

## 6. 可观测性

结构化 JSON 日志，每条带 `trace_id` 与 `session_id`。**日志走 stderr**，
stdout 留给 CLI 的 JSON 输出——这样 `sma eval > report.json` 不会被日志污染。

关键事件：`docs_loaded`、`index_done`、`bm25_built`、`retrieval_done`、
`intent_routed`、`injection_detected`、`relevance_guard_blocked`、
`tool_executed`、`tool_forbidden`、`tool_high_risk_blocked`、`citation_dropped`、`chat_done`。

`RetrievalTrace` 把候选、融合顺序、重排顺序、最终上下文都记下来，
通过 `include_trace: true` 返回——排查"为什么答成这样"时直接看它。

## 7. 扩展点

| 要替换 | 怎么做 |
|--------|--------|
| Embedding 模型 | 实现 `Embedder`，在 `llm/factory.py` 注册；**换完必须全量重建索引** |
| 向量库 | 替换 `ChromaVectorStore`（接口：upsert / query / delete_doc / all_chunks / count） |
| 重排器 | 实现 `Reranker`（如接 bge-reranker），在 `HybridRetriever` 注入 |
| 融合策略 | 改 `reciprocal_rank_fusion`，或在 `HybridRetriever` 里换成加权归一化 |
| 新业务工具 | 继承 `BaseTool`，声明权限属性，在 `build_registry` 注册 |
| 意图类型 | 加 `Intent` 枚举 → `INTENT_ROUTES` → `INTENT_TOOL_MAP` |
