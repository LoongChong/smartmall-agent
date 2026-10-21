# Agent 设计问答

## Q1：为什么用 LangGraph？不用会怎样？

这个流程的本质是**条件路由**：根据意图走完全不同的分支，分支数量固定、编译期已知。
这正是图编排的适用场景。

不用的话就是一堆 if/elif。能跑，但有两个实际损失：

1. **可审计性**。现在 `INTENT_ROUTES` 这张表集中在一处，一眼能看完
   "什么情况下会走到工具/人工"。散在控制流里就要读代码才知道。
2. **节点可单测**。每个图节点对应 `ChatService` 上的一个方法，
   `tests/test_agent.py` 直接调 `do_detect_intent`、`do_answer_from_knowledge`，
   不需要启动整个图。

**追问：那为什么你姊妹项目的子任务调度没用 LangGraph？**

判断标准是同一条：**这个流程是不是声明式的条件路由**。

| 特征 | 本项目 Agent 流程 | DRA 子任务调度 |
|------|------------------|---------------|
| 分支数量 | 固定 4 条，编译期已知 | 运行期由 Planner 动态生成 |
| 核心逻辑 | 条件路由 | 依赖拓扑 + 并发控制 + 失败传播 |
| 并发 | 无 | 需要并发度控制和就绪集合重算 |
| 持久化恢复 | 不需要 | 需要跨进程租约领取 |

第二列那些 LangGraph 都不直接提供，硬套上去会变成"用图的 API 手写一个调度器"——
既没得到框架收益，又多一层抽象。

结论不是"LangGraph 好或不好"，是**选型要看流程的本质**。

## Q2：为什么没用 CrewAI？需求里专门提到了它。

需求文档写的是"不要为了凑齐框架而强行使用 CrewAI"，所以我给出了判断依据。

CrewAI 的价值在**多个有不同角色的智能体互相协作**（研究员 → 写作者 → 审校）。
我这个流程是：一个问题 → 识别意图 → 走检索或走工具 → 生成 → 校验。
**全程只有一个决策主体**，分支之间不交换中间产物，没有角色分工，没有协作轮次。

引入它会得到一层 Agent/Task/Crew 的包装、一套额外的提示词模板约定、一个新依赖——
换不到任何功能。

更关键的是它会**损害安全性**：CrewAI 的 Agent 默认持有工具列表并自行决定调用哪个。
而我的核心安全设计恰恰是"模型不能指定工具名，意图到工具的映射是代码常量表"。
要在 CrewAI 里实现同样约束，就得绕过它的工具调度——那还用它做什么。

**什么情况下我会换**：如果需求变成"客服 Agent 处理不了的工单自动转给质检 Agent 复核，
再转给工单 Agent 生成处理方案"——那是真的多智能体协作，届时值得重新评估。

## Q3：Agent 怎么保证不无限循环？

**图里没有任何回边。** `retrieve → answer → validate → END` 是单向的，
`validate` 不会回到 `route`。

路径长度有硬上界：`route → tool → tool_answer → validate` = 4 个节点，
其中最多 2 次模型调用（意图识别 + 回答生成）。成本可预测。

另外 `max_tool_calls=3` 限制单轮工具调用次数。**但我会主动说明**：
当前实现每轮最多调 1 个工具，这个配置项还没有实际约束力，
它是为将来多工具编排预留的。预留方式是**配置项先存在**，而不是现在就写一个用不上的循环结构。

## Q4：状态设计上踩过什么坑？

**LangGraph 会静默丢弃未在 `AgentState` TypedDict 里声明的键，而且不报错。**

最初 `raw_citations`、`tool_summary`、`tool_ok`、`tool_error` 没声明。
节点返回时带上了，但下一个节点拿不到。

表现是：引用校验永远拿到空的 `raw_citations`，于是**所有知识问答都因为"没有真引用"
被降级成 `no_answer`**。看起来像是引用校验太严，实际是状态在节点之间丢了。

修复就是补全声明，并在代码里留了注释——这个行为不看文档很难想到。

**追问：状态里你刻意不放什么？**

- 会话历史的完整副本（只放最近 6 条，完整历史在 SessionStore）；
- 业务数据原件（只放 `tool_data` 摘要，不把订单对象塞进去）；
- 密钥和连接对象（在 `ChatService` 实例上）。

状态是每轮对话的临时载体，不是数据层。

## Q5：意图识别失败了怎么办？

降级为 `KNOWLEDGE_QA`：

```python
except SMAError as exc:
    logger.warning("intent_fallback", extra={"code": exc.code})
    return IntentResult(intent=Intent.KNOWLEDGE_QA, confidence=0.0, reason=f"意图识别失败，降级为知识库检索: {exc.code}")
```

**为什么这是最安全的兜底**：知识库分支**不触发任何工具调用，不做任何写操作**。
如果降级到工具分支，模型挂掉的时候反而可能误触发业务操作。

模型输出了枚举外的意图值也落到 `KNOWLEDGE_QA`，同样是保守方向。

**追问：有没有不该兜底的失败？**

有，而且这是我刻意处理的反例：

```python
except KnowledgeBaseEmptyError:
    raise                    # 不兜底
except SMAError:
    final = {..., "status": ERROR, "answer": "抱歉，服务暂时不可用…"}
```

索引为空是**配置问题**（部署时忘了跑 `sma index`），不是业务问题。
如果把它也转成"服务暂时不可用"，这个问题会被完全掩盖。它应该是 503，让运维看到。

**不是所有失败都该兜底**——该暴露的要暴露。

## Q6：工具选择权为什么不交给模型？

```python
INTENT_TOOL_MAP: dict[Intent, str] = {
    Intent.ORDER_QUERY: "order_query",
    ...
}
```

**模型输出的是意图，不是工具名。**

如果让模型直接指定 `tool_name`，它就能（被诱导）指定任意已注册工具，包括 `refund_order`。
现在它最多只能表达"我认为这是退款意图"，而退款意图被路由到 `escalate`（转人工），
**永远到不了那个工具**。

这是"能力最小化"的具体落地：把"模型被攻破"当作既定事实来设计，然后限制损失上界。

## Q7：工具调用的框架是怎么设计的？

`BaseTool.execute()` 五段式，任何工具调用都必须经过它：

```
① 参数校验   Pydantic model_validate        → INVALID_ARGS
② 鉴权       requires_auth & ctx            → FORBIDDEN
③ 高风险拦截 is_high_risk & !approved       → NEEDS_APPROVAL
④ 带超时执行 asyncio.timeout → _run         → TIMEOUT
⑤ 异常分类   Forbidden/SMAError/Exception   → FORBIDDEN/FAILED
```

子类只实现 `_run`。**每一段都产出 `ToolCallRecord`**——
所以无论在哪一段被拦截，都有一条审计记录（工具名、状态、脱敏参数、错误码、耗时、结果摘要）。

**追问：为什么参数校验要在业务逻辑之前？**

参数来自模型输出，而模型输出受用户输入影响。
`max_length` 防止超长字符串塞进日志和查询；字符白名单把注入类输入挡在业务层之外。

```python
order_id: str = Field(min_length=4, max_length=40)


@field_validator("order_id")
def _strip(cls, v):
    if not v.replace("-", "").isalnum():
        raise ValueError("订单号只能包含字母、数字和连字符")
```

**追问：第 5 段为什么要兜底捕获 Exception？**

一个未预期的 `KeyError` 不应该变成 HTTP 500 和一段堆栈给用户看。
所以兜底捕获，但必须满足三件事：`logger.exception` 记全量、返回明确的失败状态、
`# noqa: BLE001` 旁边写明理由。空 except 是禁止的。

## Q8：槽位缺失怎么处理？

```python
missing = [s for s in TOOL_REQUIRED_SLOTS.get(tool_name, ()) if not intent_result.slots.get(s)]
if missing:
    return {"status": CLARIFY, "answer": CLARIFY_TEMPLATE.format(missing=labels)}
```

走澄清，**不让工具白跑一次再报错**。

必填槽位声明在 `TOOL_REQUIRED_SLOTS` 常量表里，和工具的 `args_schema` 是两层：
schema 管"格式对不对"，这张表管"该不该现在调"。

## Q9：模型挂了，订单查询还能用吗？

能。这是刻意设计的：

```python
try:
    response = await self.llm.complete(messages, purpose="tool_answer")
    text = response.content.strip()
except (SMAError, LLMError):
    logger.warning("tool_answer_fallback", ...)
    text = summary  # 直接用工具摘要
```

**工具返回的数据是确定性的，不依赖模型。** 模型在这个节点的唯一作用是把
`订单 SO20261001001：星云 Pro 14 ×1，金额 5999.0 元，当前状态 delivered`
说得更口语化。它挂了就直接把摘要给用户——信息完整，只是生硬一点。

这比"模型不可用 → 整个查询失败"好得多。

## Q10：业务结论为什么必须在代码里？

**因为模型会被说服。**

用户说"我真的很需要退货，求你了"，模型有相当概率改口说"可以退"。
`check_return_eligibility()` 不会。

所以：退货资格 → 确定性函数；订单归属 → `assert_order_ownership()`；
高风险操作 → 框架层拦截；写操作 → 写之前**重跑一遍规则**。

`TOOL_ANSWER_SYSTEM_PROMPT` 里也写了"绝不自行判断用户是否可以退货/退款"，
但那只是降低概率的辅助，不是安全控制。

**追问：规则判定为什么要收集全部原因而不是短路返回？**

```python
reasons: list[str] = []
...
return ReturnEligibility(eligible=not reasons, reasons=reasons, ...)
```

用户体验。短路返回会让用户修一个问题、再试一次、又发现另一个问题。
一次把全部障碍说清楚，用户一次就知道能不能退。

而且返回结构里带 `policy_days`、`days_since_delivery`、`deadline`——
不只给结论还给依据，前端可以展示"签收于 10-01，截止 10-08，已过期 7 天"。

## Q11：写操作怎么保证幂等？

```python
key = sha256(f"{ctx.user_id}|{order.order_id}|{args.reason.strip()}").hexdigest()[:16]
if existing_id := _IDEMPOTENCY.get(key):
    raise DuplicateSubmissionError(f"该订单已存在相同原因的售后申请（工单号 {ticket.ticket_id}），请勿重复提交")
```

键是 `用户 + 订单 + 原因`。**为什么包含原因**：同一订单可能有多个不同诉求
（"屏幕有划痕"和"充电器不工作"应该是两张工单），但同一原因重复提交就是重复。

重复时抛错并**告知已有工单号**，不静默返回成功——用户需要工单号去查进度。

**我会主动说的缺陷**：这个幂等存储是**进程内字典**。
进程重启丢失，多实例部署不共享。生产必须落库，幂等键建唯一索引，
用 `INSERT ... ON CONFLICT DO NOTHING` + `rowcount` 判定。已记在 ROADMAP。

## Q12：写操作前为什么要重跑规则？前面不是已经预检查过了吗？

因为**预检查和实际创建之间可能隔了几分钟，订单状态可能已经变了**。

更重要的是：不能因为模型（或前一次预检查）说"可以退"就创建工单。
**写操作自己负责校验自己的前置条件**，不信任上游的结论。

## Q13：会话管理怎么做的？

SQLite 两张表（`sessions` + `turns`），三个设计点：

1. **绑定 user_id**。`history()` 读取时强制校验归属——知道 session_id 也读不了别人的会话，
   直接 `ForbiddenError` 而不是返回空列表。
2. **历史有上限**。`session_max_turns=10`（20 条 turn），超出丢弃最旧的。
   防止单个会话无界增长成为内存/存储攻击面。
3. **失败的轮次也写入历史**。否则上下文会断。

**追问：这里有已知问题吗？**

有。seq 是用 `SELECT MAX(seq)+1` 算的，同一会话并发写入时可能冲突
（有 PRIMARY KEY 兜底会报错，不会写坏数据，但请求会失败）。
正确做法是数据库自增或乐观锁。已记在 ROADMAP 的已知缺陷表里。

## Q14：怎么加一个新意图？

1. `models.py` 加 `Intent` 枚举值；
2. `prompts.py:INTENT_SYSTEM_PROMPT` 的意图清单加一行描述；
3. `workflow.py:INTENT_ROUTES` 加路由；
4. 走工具的话：`INTENT_TOOL_MAP` 加映射，`TOOL_REQUIRED_SLOTS` 声明必填槽位；
5. **`eval/dataset.py` 加至少一条行为用例**；
6. 跑 `uv run sma eval` 确认没回归。

**第 5 步不是可选的。** 加意图最容易出的问题是**抢了已有意图的分类**——
比如新增"修改收货地址"会抢走一部分"订单查询"。只有评测集能发现这个。

## Q15：可观测性做到什么程度？

结构化 JSON 日志，每条带 `trace_id`（contextvars 透传）和 `session_id`。
`trace_id` 可以从请求头 `X-Trace-Id` 传入，也会回写到响应头。

**日志走 stderr**，stdout 留给 CLI 的 JSON 输出——这样 `sma eval > report.json` 不会被污染。
（这个是被 jieba 的词典加载进度打到 stdout 污染过才改的。）

关键事件分三类：

- **正常流程**：`intent_routed`、`retrieval_done`、`tool_executed`、`chat_done`
- **防护生效**：`injection_detected`、`relevance_guard_blocked`、`tool_forbidden`、
  `tool_high_risk_blocked`、`citation_dropped`、`session_owner_mismatch`
- **要查的异常**：`intent_fallback`、`tool_answer_fallback`、`tool_timeout`、
  `chat_crashed`、`startup_index_failed`

第二类的**频率**本身就是质量指标：`citation_dropped` 占比突然上升说明提示词约束失效了；
`relevance_guard_blocked` 太高说明阈值在误杀。我在 RELEASE 文档里给了建议的告警阈值。
