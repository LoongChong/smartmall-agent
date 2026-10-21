# Agent 工作流（LangGraph）

## 1. 图结构

```
                        ┌─→ retrieve → answer ──────┐
                        │                            │
  (entry) route ────────┼─→ tool ────→ tool_answer ──┼─→ validate ─→ END
                        │                            │
                        ├─→ clarify ─────────────────────────────────→ END
                        └─→ escalate ────────────────────────────────→ END
```

8 个节点，**没有回边**。每条路径最多经过 4 个节点（route → tool → tool_answer → validate）
就到 END。这是刻意的——见第 4 节。

## 2. 为什么这里用 LangGraph

这个项目里只有 Agent 主流程用了 LangGraph，而姊妹项目 Deep Research Agent 的
子任务调度**没有**用（它自己实现了 DAG 调度器）。判断标准是同一条：

| 特征 | 本项目的 Agent 流程 | DRA 的子任务调度 |
|------|---------------------|------------------|
| 分支数量 | 固定 4 条，编译期已知 | 运行期由 Planner 动态生成 |
| 核心逻辑 | **条件路由** | 依赖拓扑 + 并发控制 + 失败传播 |
| 并发 | 无（单轮串行） | 需要并发度控制和就绪集合重算 |
| 持久化恢复 | 不需要（单轮无状态） | 需要（跨进程租约领取） |

LangGraph 擅长的正是第一列：**声明式的条件路由 + 状态传递**。
用它写出来的 `add_conditional_edges` 比一堆 if/elif 更能直接读出"什么情况下会走到哪"，
这对审计能力边界有实际价值。

而第二列那些需求 LangGraph 都不直接提供，硬套上去会变成"用图的 API 手写一个调度器"，
既没得到框架收益，又多一层抽象。所以那边自己写了。

**结论不是"LangGraph 好/不好"，而是"这个流程是不是条件路由"。** 是，就用它。

## 3. 状态设计

```python
class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str | None
    authenticated: bool
    question: str
    history: list[dict[str, str]]

    intent: IntentResult
    retrieved: list[RetrievedChunk]
    retrieval_trace: RetrievalTrace | None

    tool_calls: list[ToolCallRecord]
    tool_data: dict[str, Any] | None
    tool_summary: str
    tool_ok: bool
    tool_error: str | None

    answer: str
    raw_citations: list[str]  # 模型原始输出，未经校验
    status: AnswerStatus
    citations: list[Citation]  # 校验后的真实引用
    dropped_citations: list[str]
    needs_human: bool
```

### 踩过的坑：LangGraph 会静默丢弃未声明的键

最初 `raw_citations`、`tool_summary`、`tool_ok`、`tool_error` 没写进 TypedDict，
节点返回时带上了它们，但**下一个节点拿不到**——LangGraph 按 schema 过滤状态，
未声明的键被丢掉，而且不报错。

表现是：引用校验永远拿到空的 `raw_citations`，于是所有知识问答都因为
"没有真引用"被降级成 `no_answer`。看起来像是引用校验太严，实际是状态丢了。

修复就是补全声明，并在代码里留了注释，因为这个行为不看文档很难想到：

```python
    # 注意：LangGraph 会丢弃未在 TypedDict 中声明的键，所有跨节点传递的字段都必须在这里列出。
```

### 状态里不放什么

- **会话历史的完整副本**：只放最近 6 条（`history[-6:]`），完整历史在 SessionStore。
- **业务数据原件**：工具返回的数据只放 `tool_data` 摘要，不把订单对象塞进状态。
- **密钥、连接对象**：这些在 `ChatService` 实例上，不进状态。

状态是每轮对话的临时载体，不是数据层。

## 4. 有限循环：为什么没有回边

Agent 最常见的失控方式是"自己和自己聊下去"：模型说"我需要再查一次"，
于是又回到工具节点，再查，再说需要查……直到超时或者烧掉几十次模型调用。

这个图从结构上杜绝了它：**图里没有任何回边**。
`retrieve → answer → validate → END` 是单向的，`validate` 不会回到 `route`。

另外 `max_tool_calls=3`（可配，上限 10）限制单轮工具调用次数。
当前实现每轮最多调 1 个工具，这个上限是为将来多工具编排预留的。
预留的方式是**配置项先存在**，而不是现在就写一个用不上的循环结构。

路径长度上界：`route(1) → tool(1) → tool_answer(1) → validate(1)` = 4 个节点，
其中最多 2 次模型调用（意图识别 + 回答生成）。成本可预测。

## 5. 各节点职责

每个图节点都是 `ChatService` 上的一个方法，**方法本身可以脱离 LangGraph 单测**
（`tests/test_agent.py` 直接调 `do_detect_intent`、`do_answer_from_knowledge` 等）。
图只负责装配和路由。

### route：意图识别

```python
async def do_detect_intent(self, state) -> IntentResult:
    if pattern := detect_injection(question):  # ① 确定性短路，先于模型
        return IntentResult(intent=Intent.OUT_OF_SCOPE, confidence=1.0, reason=f"命中注入模式: {pattern}")
    detect_identity_spoof(question)  # ② 只告警不拦截
    ...
    output = await self.llm.complete_json(messages, IntentOutput, purpose="intent")
    ...
    slots = {k: v for k, v in output.slots.items() if k not in {"user_id", "uid", "account"}}
```

三个关键点：

**① 注入检测在模型之前。** 提示词里写"遇到注入请归类为 out_of_scope"是概率性的；
模式匹配是确定性的。命中即短路，根本不进入检索和工具分支。
它会漏报变体表达，但不会被"请忽略上面的规则"这类直白攻击绕过。

**② 身份冒充只告警。** 用户说"我是管理员"可能只是随口一说，拒绝服务是过度反应。
真正的保护在别处：鉴权永远用 `ToolContext.user_id`（来自认证层），
聊天内容影响不了它。这里记日志是为了能发现攻击尝试。

**③ 模型给出的 slots 里过滤掉身份字段。** 即使模型被诱导输出
`slots: {"user_id": "U002"}`，这个键也会被丢掉。**身份只有一个来源**。

意图识别失败时的兜底是 `KNOWLEDGE_QA`：

```python
except SMAError as exc:
    return IntentResult(intent=Intent.KNOWLEDGE_QA, confidence=0.0, reason=f"意图识别失败，降级为知识库检索: {exc.code}")
```

为什么这是最安全的兜底：知识库分支**不触发任何工具调用，不做任何写操作**。
如果降级到工具分支，模型挂掉的时候反而可能误触发业务操作。

### 路由表

```python
INTENT_ROUTES: dict[Intent, str] = {
    Intent.KNOWLEDGE_QA: "retrieve",
    Intent.PRODUCT_QUERY: "tool",
    Intent.ORDER_QUERY: "tool",
    Intent.LOGISTICS_QUERY: "tool",
    Intent.RETURN_ELIGIBILITY: "tool",
    Intent.AFTERSALE_CREATE: "tool",
    Intent.HIGH_RISK: "escalate",
    Intent.CLARIFY: "clarify",
    Intent.OUT_OF_SCOPE: "escalate",
}
```

集中在一处，一眼能看完"什么情况下会走到工具/人工"。这是能力边界的可审计性。

未知意图（模型输出了枚举外的字符串）落到 `KNOWLEDGE_QA`，同样是保守方向。

### retrieve / answer

见 [RAG_PIPELINE](RAG_PIPELINE.md)。关键是 `answer` 节点在调模型前先过相关性门槛。

### tool：工具选择权不在模型手里

```python
INTENT_TOOL_MAP: dict[Intent, str] = {
    Intent.PRODUCT_QUERY: "product_query",
    Intent.ORDER_QUERY: "order_query",
    Intent.LOGISTICS_QUERY: "logistics_query",
    Intent.RETURN_ELIGIBILITY: "return_eligibility_check",
    Intent.AFTERSALE_CREATE: "aftersale_create",
}
```

**模型输出的是意图，不是工具名。** 意图到工具的映射是代码里的常量表。

这样做的理由：如果让模型直接指定 `tool_name`，它就能（被诱导）指定任意已注册工具，
包括 `refund_order`。现在它最多只能表达"我认为这是退款意图"，
而退款意图被路由到 `escalate`（转人工），永远到不了 `RefundTool`。

槽位缺失走澄清，不让工具白跑：

```python
missing = [s for s in TOOL_REQUIRED_SLOTS.get(tool_name, ()) if not intent_result.slots.get(s)]
if missing:
    return {"status": CLARIFY, "answer": CLARIFY_TEMPLATE.format(missing=labels)}
```

工具返回的三种终态会被直接转成回答，不再进模型：

| 工具状态 | 回答状态 | 话术 |
|---------|---------|------|
| `FORBIDDEN` | `FORBIDDEN` | "未找到该订单，或该订单不属于当前登录账号" |
| `NEEDS_APPROVAL` | `ESCALATED` + `needs_human=True` | 高风险转人工话术 |
| 其他失败 | `NO_ANSWER` | 工具的错误消息 + 建议联系人工 |

### tool_answer：模型只负责说人话

```python
try:
    response = await self.llm.complete(messages, purpose="tool_answer")
    text = response.content.strip()
except (SMAError, LLMError):
    text = summary  # 模型挂了直接用工具摘要
```

这个 fallback 很关键：**工具返回的数据是确定性的，不依赖模型也能回答**。
模型在这里的唯一作用是把 `订单 SO20261001001：星云 Pro 14 ×1，金额 7999.0 元，当前状态 delivered`
说得更口语化。它挂了就直接把摘要给用户——信息完整，只是生硬一点。

这比"模型不可用 → 整个查询失败"好得多。

`TOOL_ANSWER_SYSTEM_PROMPT` 里有一条专门的约束：
**"绝不自行判断用户是否可以退货/退款——这类结论只能来自工具返回的字段"**。

### validate

```python
def do_validate(self, state) -> dict[str, Any]:
    if not retrieved:
        return {}  # 工具分支没有知识库引用，跳过
    new_status, citations, dropped = validate_citations(...)
```

工具分支也经过这个节点，但因为 `retrieved` 为空直接返回。
这样图结构更简单（两条主链路汇聚到同一个终点），代价只是一次空判断。

### clarify / escalate

纯同步方法，不调模型（话术是模板）。`escalate` 区分两种情况：
`HIGH_RISK` → `needs_human=True`（要人接手），`OUT_OF_SCOPE` → `needs_human=False`
（只是超出范围，不需要人工介入）。

## 6. 异常不击穿到用户

`ChatService.chat` 外层三层兜底：

```python
try:
    final = await self._graph.ainvoke(state)
except KnowledgeBaseEmptyError:
    raise                       # ① 索引为空必须暴露，这是配置问题不是业务问题
except SMAError as exc:
    final = {..., "status": ERROR, "answer": "抱歉，服务暂时不可用，请稍后重试或联系人工客服。"}
except Exception:
    logger.exception("chat_crashed")
    final = {..., "status": ERROR, "answer": "抱歉，处理您的问题时出现异常，已记录并将尽快修复。"}
```

① 单独放行 `KnowledgeBaseEmptyError`：如果把它也转成"服务暂时不可用"，
部署时忘了跑 `sma index` 这个问题会被完全掩盖。它应该是 503，让运维看到。

② 任何异常都不把堆栈、错误码、内部字段名暴露给用户（`logger.exception` 记全量，
返回给用户的是脱敏话术）。

③ **无论成功失败，都会写入会话历史**，并记录 `chat_done` 日志。
失败的轮次也要留在历史里，否则上下文会断。

## 7. 可观测性

每次对话产生一条 `chat_done`，带 `trace_id` / `session_id` / `intent` / `status` /
`citations` 数量 / `tools` 数量 / `elapsed_ms`。

中间事件：`injection_detected`、`identity_spoof_attempt`、`intent_routed`、
`intent_fallback`、`relevance_guard_blocked`、`no_answer_by_guard`、
`tool_executed`、`tool_forbidden`、`tool_high_risk_blocked`、`citation_dropped`、
`answer_without_citation`、`unsupported_numbers`、`tool_answer_fallback`。

日志走 **stderr**，stdout 留给 CLI 的 JSON 输出（`sma eval > report.json` 不会被污染）。

## 8. 怎么加一个新意图

1. `models.py` 加 `Intent` 枚举值；
2. `prompts.py:INTENT_SYSTEM_PROMPT` 的意图清单里加一行描述；
3. `workflow.py:INTENT_ROUTES` 加路由；
4. 如果走工具：`service.py:INTENT_TOOL_MAP` 加映射，`TOOL_REQUIRED_SLOTS` 声明必填槽位；
5. `eval/dataset.py` 加至少一条行为用例；
6. 跑 `uv run sma eval` 确认没有回归。

第 5 步不是可选的。加意图最容易出的问题是抢了已有意图的分类
（比如新增"修改收货地址"会抢走一部分"订单查询"），只有评测集能发现。
