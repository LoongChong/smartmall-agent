"""业务规则引擎。

**退货资格由这里的确定性代码判定，不由模型判定。**
这是整个项目最重要的安全边界之一：如果让 LLM 说"你可以退货"，
它会因为用户一句"但是我真的很需要退"就改口。规则引擎不会。

规则来源与知识库文档 `aftersale_return_policy.md` 保持一致；
两者不一致时**以代码为准**，并在文档里说明如何同步。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..models import Order, OrderStatus, Product, ReturnEligibility
from .mock_data import NOW

#: 不支持 7 天无理由的商品类型标记
NON_RETURNABLE_REASON = "该商品为定制类商品，不适用 7 天无理由退货"


def check_return_eligibility(
    order: Order, product: Product | None, *, now: datetime | None = None
) -> ReturnEligibility:
    """判定订单是否满足 7 天无理由退货条件。

    判定顺序（任一不满足即 eligible=False，但**所有**原因都会被收集，
    这样用户一次就能知道全部障碍，不用反复试）：

    1. 订单状态必须是已签收（未签收无法退货，应走取消订单流程）
    2. 商品必须支持无理由退货（定制商品 return_days=0）
    3. 签收后天数必须在政策天数内
    4. 订单不能已在退款流程中
    """
    now = now or NOW
    policy_days = product.return_days if product else 7
    reasons: list[str] = []
    days_since: int | None = None
    deadline: datetime | None = None

    if order.status == OrderStatus.CANCELLED:
        reasons.append("订单已取消，无需退货")
    elif order.status == OrderStatus.REFUNDING:
        reasons.append("订单已在退款流程中，请勿重复申请")
    elif order.status != OrderStatus.DELIVERED or order.delivered_at is None:
        reasons.append(f"订单当前状态为「{_status_cn(order.status)}」，尚未签收，暂不支持退货申请")

    if policy_days <= 0:
        reasons.append(NON_RETURNABLE_REASON)

    if order.delivered_at is not None and policy_days > 0:
        days_since = (now - order.delivered_at).days
        deadline = order.delivered_at + timedelta(days=policy_days)
        if days_since > policy_days:
            reasons.append(
                f"签收已满 {days_since} 天，超出 {policy_days} 天无理由退货期限（截止时间 {deadline:%Y-%m-%d}）"
            )

    return ReturnEligibility(
        order_id=order.order_id,
        eligible=not reasons,
        reasons=reasons,
        policy_days=policy_days,
        days_since_delivery=days_since,
        deadline=deadline,
    )


def _status_cn(status: OrderStatus) -> str:
    return {
        OrderStatus.PENDING_PAYMENT: "待付款",
        OrderStatus.PAID: "已付款待发货",
        OrderStatus.SHIPPED: "运输中",
        OrderStatus.DELIVERED: "已签收",
        OrderStatus.CANCELLED: "已取消",
        OrderStatus.REFUNDING: "退款中",
    }[status]


def assert_order_ownership(order: Order, user_id: str | None) -> None:
    """订单归属校验。

    **这个函数是防越权的唯一入口**，所有涉及订单的工具都必须先调用它。
    user_id 必须来自已认证的会话，绝不接受用户在聊天中自述的身份。
    """
    from ..errors import ForbiddenError

    if not user_id:
        raise ForbiddenError("无法确认您的身份，请先登录后再查询订单")
    if order.user_id != user_id:
        # 注意：不泄漏"订单存在但不属于你"这个信息差异，统一话术
        raise ForbiddenError("未找到该订单，或该订单不属于当前登录账号")
