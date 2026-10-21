# 工具契约与业务规则

## 0. 一句话原则

> **LLM 只负责识别意图和抽取参数；是否允许执行、业务结论是什么，全部由后端代码决定。**

这条原则贯穿本文档的每一节。它不是风格偏好，而是因为：
模型会被说服。用户说"我真的很需要退货，求你了"，模型有相当概率改口说"可以退"。
规则引擎不会。

## 1. 工具框架：五段式执行

`tools/base.py:BaseTool.execute()` 是唯一入口，任何工具调用都必须经过它：

```
raw_args, ctx
   │
   ├─ ① 参数校验   args_schema.model_validate   → INVALID_ARGS
   ├─ ② 鉴权       requires_auth & ctx          → FORBIDDEN
   ├─ ③ 高风险拦截 is_high_risk & !approved     → NEEDS_APPROVAL
   ├─ ④ 带超时执行 asyncio.timeout(timeout)     → TIMEOUT
   │     └─ _run(args, ctx)                     ← 子类只实现这一个方法
   └─ ⑤ 异常分类   Forbidden / SMAError / 其他  → FORBIDDEN / FAILED
   │
   ▼
(ToolResult, ToolCallRecord)
```

每一段都产出 `ToolCallRecord`（工具名、状态、脱敏参数、错误码、耗时、结果摘要），
所以**无论在哪一段被拦截，都有一条审计记录**。

### ① 参数校验

```python
args = self.args_schema.model_validate(raw_args or {})
```

Pydantic 模型在进入业务逻辑**之前**完成校验。每个工具声明自己的 schema：

```python
class OrderQueryArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)

    @field_validator("order_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("-", "").isalnum():
            raise ValueError("订单号只能包含字母、数字和连字符")
        return v
```

长度上限和字符白名单不是形式主义：参数来自模型输出，而模型输出受用户输入影响。
`max_length` 防止超长字符串塞进日志和查询；字符白名单把注入类输入挡在业务层之外。

### ② 鉴权：身份只能来自认证层

```python
if self.requires_auth and not (ctx.authenticated and ctx.user_id):
    return FORBIDDEN
```

```python
@dataclass(slots=True)
class ToolContext:
    user_id: str | None  # 来自已认证的会话，不是聊天内容
    session_id: str
    authenticated: bool = False
    approved_high_risk: bool = False
```

`ToolContext` 由 `ChatService.do_call_tool` 构造，`user_id` 取自 `state["user_id"]`，
而那个值来自 HTTP 的 `X-User-Id` 头（由网关/认证层注入）。

**从聊天内容到 `ctx.user_id` 没有任何通路。** 意图识别那一层还额外过滤了
模型输出的 `user_id` / `uid` / `account` 槽位。

### ③ 高风险拦截

```python
if self.is_high_risk and not ctx.approved_high_risk:
    return NEEDS_APPROVAL
```

`approved_high_risk` 当前**没有任何代码路径会把它设为 True**。
这是故意的：本项目不实现审批流，所以高风险工具必然被拦截。
这比"留一个后门参数但没人用"更诚实——如果将来接审批系统，
就在构造 `ToolContext` 时根据审批结果传入。

### ④ 超时

```python
async with asyncio.timeout(timeout):  # timeout 来自 settings.tool_timeout_seconds，默认 10s
    result = await self._run(args, ctx)
```

用 Python 3.11 的 `asyncio.timeout`。

`execute()` 的 `timeout` 参数带了 `# noqa: ASYNC109` 并写明理由：
ruff 建议用 `asyncio.timeout` 替代 timeout 参数，但这里 timeout 是**工具的业务配置项**
（需要从 settings 注入、需要能按工具覆盖），内部确实是用 `asyncio.timeout` 实现的。
规则的意图已经满足，所以抑制告警而不是改设计。

### ⑤ 异常分类

```python
except TimeoutError:        → TIMEOUT,   "查询超时，请稍后重试"
except ForbiddenError:      → FORBIDDEN, "无权访问该资源"
except SMAError as exc:     → FAILED,    exc.message[:200]
except Exception:           → FAILED,    "工具执行异常"   + logger.exception
```

最后那个 `except Exception` 带 `# noqa: BLE001`，理由写在代码里：
**工具异常不得击穿到用户**。一个未预期的 `KeyError` 不应该变成 500 和一段堆栈。

## 2. 工具清单

| 工具 | 鉴权 | 写 | 高风险 | 说明 |
|------|:----:|:--:|:------:|------|
| `product_query` | — | — | — | 商品信息是公开的，不需要身份 |
| `order_query` | ✓ | — | — | 订单是用户私有数据 |
| `logistics_query` | ✓ | — | — | 按订单号或运单号查，两条路径都校验归属 |
| `return_eligibility_check` | ✓ | — | — | 结论由规则引擎给出 |
| `aftersale_create` | ✓ | ✓ | — | 写操作，带幂等键 |
| `refund_order` | ✓ | ✓ | **✓** | **永远被框架拦截** |

能力边界可审计：`GET /api/v1/tools`（需管理员令牌）返回每个工具的
`requires_auth` / `is_write` / `is_high_risk` 和完整 JSON Schema。

### 为什么保留一个永远不执行的 RefundTool

它展示的是高风险操作的**完整处理路径**：

```
用户："帮我把这单退款"
  → 意图识别 → Intent.HIGH_RISK
  → INTENT_ROUTES → escalate 节点
  → 转人工，needs_human=True
```

注意它甚至走不到 `tool` 节点——`HIGH_RISK` 意图被路由到 `escalate`。
即使绕过意图路由（比如将来有人把它加进 `INTENT_TOOL_MAP`），
`BaseTool.execute` 的第 ③ 段还会拦一次。两层防护。

而且 `_run` 里是这样写的：

```python
async def _run(self, args, ctx) -> ToolResult:
    raise ToolError("退款能力尚未对接支付系统，请转交人工客服处理")
```

**不假装成功。** 没对接支付系统就如实抛错，而不是返回一个 `{"ok": true, "refunded": 7999}`
的假结果。

## 3. 业务规则引擎

### 3.1 退货资格判定

`tools/business_rules.py:check_return_eligibility()` 是确定性函数：

```python
def check_return_eligibility(order, product, *, now=None) -> ReturnEligibility
```

判定顺序：

1. 订单状态必须是**已签收**（未签收应走取消订单流程，不是退货）；
2. 商品必须支持无理由退货（定制类商品 `return_days=0`）；
3. 签收后天数必须在政策天数内；
4. 订单不能已在退款流程中。

**所有不满足的原因都会被收集，不是遇到第一个就返回：**

```python
reasons: list[str] = []
...
return ReturnEligibility(order_id=..., eligible=not reasons, reasons=reasons, ...)
```

理由是用户体验：短路返回会让用户修一个问题、再试一次、又发现另一个问题。
一次把全部障碍说清楚，用户一次就知道能不能退。

返回结构里包含 `policy_days`、`days_since_delivery`、`deadline`——
不只给结论，还给依据。前端可以展示"签收于 10-01，截止 10-08，已过期 7 天"。

### 3.2 订单归属校验

```python
def assert_order_ownership(order: Order, user_id: str | None) -> None:
    if not user_id:
        raise ForbiddenError("无法确认您的身份，请先登录后再查询订单")
    if order.user_id != user_id:
        raise ForbiddenError("未找到该订单，或该订单不属于当前登录账号")
```

**这是防越权的唯一入口**，所有涉及订单的工具（`order_query`、`logistics_query`、
`return_eligibility_check`、`aftersale_create`）都必须先调它。

注意话术：订单不存在和订单不属于你，返回**完全相同**的消息。

```python
# order_query 里，订单不存在时：
error_message = "未找到该订单，或该订单不属于当前登录账号"
```

不这样做的后果：攻击者遍历订单号，根据"订单不存在"和"无权访问"的区别
就能枚举出系统里哪些订单号是真实的。这是典型的信息泄漏侧信道。

`logistics_query` 还有一条容易漏的路径——直接用运单号查：

```python
owned = any(o.tracking_no == tracking_no and o.user_id == ctx.user_id for o in ORDERS.values())
if not owned:
    raise ForbiddenError("未找到该运单，或该运单不属于当前登录账号")
```

运单号也是私有数据。如果只校验"通过订单号查"这条路，攻击者换成运单号就绕过了。
**每条数据访问路径都要独立校验**，不能假设用户只走主路径。

### 3.3 规则与知识库文档的一致性

`data/knowledge/aftersale_return_policy.md` 里也写了 7 天无理由退货的规则。
两处描述同一件事，必然有不一致的风险。

约定：**不一致时以代码为准**。知识库文档负责回答"政策是什么"（给人看），
规则引擎负责判定"这一单能不能退"（给系统用）。

同步方式是人工的——改了规则就要改文档。没有做自动校验，
这是一个已知的工程债，记在 [ROADMAP](ROADMAP.md) 里。
真正的解法是把政策参数（天数、例外商品类型）抽成单一数据源，
代码和文档都从它生成，但在当前规模下这个抽象的收益不足以抵消复杂度。

## 4. 写操作：幂等

`aftersale_create` 是唯一的写操作。两道保护：

**① 写之前重跑规则：**

```python
eligibility = check_return_eligibility(order, PRODUCTS.get(order.product_id))
if not eligibility.eligible:
    raise BusinessRuleError("该订单不满足售后申请条件：" + "；".join(eligibility.reasons))
```

不能因为模型（或前一次预检查）说"可以退"就创建工单。
预检查和实际创建之间可能隔了几分钟，订单状态可能已经变了。
**写操作自己负责校验自己的前置条件。**

**② 幂等键：**

```python
key = sha256(f"{ctx.user_id}|{order.order_id}|{args.reason.strip()}").hexdigest()[:16]
if existing_id := _IDEMPOTENCY.get(key):
    raise DuplicateSubmissionError(f"该订单已存在相同原因的售后申请（工单号 {ticket.ticket_id}），请勿重复提交")
```

键是 `用户 + 订单 + 原因` 的哈希。为什么包含原因：同一订单可能有多个不同的售后诉求
（"屏幕有划痕"和"充电器不工作"应该是两张工单）；但同一原因重复提交就是重复。

重复时抛错并**告知已有工单号**，而不是静默返回成功——用户需要知道工单号去查进度。

> **这个幂等存储是进程内字典（`_TICKETS` / `_IDEMPOTENCY`）。**
> 进程重启丢失，多实例部署不共享。生产必须落库（唯一索引建在幂等键上）。
> 这是演示级实现，明确记在 [ROADMAP](ROADMAP.md) 和代码注释里。

## 5. 模拟数据

`tools/mock_data.py` 里的商品、订单、物流全部是模拟数据，**不连接任何真实系统**。

关键设计：**固定时间基准**。

```python
NOW = datetime(2026, 10, 16, ...)  # 模块级常量
check_return_eligibility(order, product, now=None)  # now 默认取 NOW
```

如果用 `datetime.now()`，"签收 5 天的订单能退货"这条测试会在 3 天后失败——
因为订单的签收时间是写死的，而"现在"在往前走。固定基准让这类时间相关的断言永远可复现。

真正接入生产时，`now` 参数留着注入真实时间，测试仍可注入固定值。

## 6. 怎么加一个新工具

```python
class MyArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)


class MyTool(BaseTool[MyArgs]):
    name = "my_tool"
    description = "一句话说清它做什么、有什么限制"
    args_schema = MyArgs
    requires_auth = True  # 碰用户私有数据就必须 True
    is_write = False
    is_high_risk = False

    async def _run(self, args: MyArgs, ctx: ToolContext) -> ToolResult:
        order = ORDERS.get(_norm_id(args.order_id))
        if order is None:
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="order_not_found",
                error_message="未找到该订单，或该订单不属于当前登录账号",  # 统一话术
                summary="订单不存在",
            )
        assert_order_ownership(order, ctx.user_id)  # 碰订单就必须调
        ...
        return ToolResult(ok=True, data={...}, summary="给模型看的人类可读摘要")
```

然后在 `build_registry()` 里注册，在 `INTENT_TOOL_MAP` 里加映射，
在 `TOOL_REQUIRED_SLOTS` 里声明必填槽位，在 `eval/dataset.py` 里加用例。

检查清单：

- [ ] 碰用户私有数据 → `requires_auth = True` **且** 调用 `assert_order_ownership`
      （两者都要，`requires_auth` 只保证"有身份"，不保证"是这条数据的主人"）
- [ ] 资源不存在的话术与越权话术**完全一致**
- [ ] 写操作 → 幂等键 + 写前重跑业务规则
- [ ] 不可逆操作 → `is_high_risk = True`
- [ ] `summary` 是给模型看的人类可读文本，不是 JSON dump
- [ ] 业务结论（能不能退、够不够库存）在 `_run` 里算完，不留给模型判断
- [ ] 评测集加用例，跑 `uv run sma eval` 确认不回归
