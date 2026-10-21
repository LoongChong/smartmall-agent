"""业务工具实现。

全部基于 mock_data 中明确标注的模拟数据，不连接真实客户、订单或支付系统。
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..errors import BusinessRuleError, DuplicateSubmissionError, ToolError
from ..logging_setup import get_logger
from ..models import AftersaleTicket, ToolStatus, utcnow
from .base import BaseTool, ToolContext, ToolResult
from .business_rules import assert_order_ownership, check_return_eligibility
from .mock_data import LOGISTICS, ORDERS, PRODUCTS

logger = get_logger(__name__)


def _norm_id(value: str) -> str:
    return value.strip().upper()


# ====================== 商品查询 ======================


class ProductQueryArgs(BaseModel):
    product_id: str | None = Field(default=None, max_length=32)
    keyword: str | None = Field(default=None, max_length=50)

    @field_validator("product_id", "keyword")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if v and v.strip() else None


class ProductQueryTool(BaseTool[ProductQueryArgs]):
    name = "product_query"
    description = "按商品 ID 或关键词查询商品的价格、库存与参数"
    args_schema = ProductQueryArgs
    requires_auth = False  # 商品信息是公开的

    async def _run(self, args: ProductQueryArgs, ctx: ToolContext) -> ToolResult:
        if not args.product_id and not args.keyword:
            raise ToolError("请提供商品 ID 或商品名称关键词")

        if args.product_id:
            product = PRODUCTS.get(_norm_id(args.product_id))
            matches = [product] if product else []
        else:
            kw = args.keyword or ""
            matches = [p for p in PRODUCTS.values() if kw in p.name or kw in p.category]

        if not matches:
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="product_not_found",
                error_message="未找到匹配的商品",
                summary="商品库中没有匹配项",
            )

        items = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "price": p.price,
                "stock": p.stock,
                "in_stock": p.stock > 0,
                "attributes": p.attributes,
                "return_days": p.return_days,
            }
            for p in matches[:5]
        ]
        summary = "；".join(
            f"{i['name']}（{i['product_id']}）售价 {i['price']} 元，{'有货' if i['in_stock'] else '暂时缺货'}"
            for i in items
        )
        return ToolResult(ok=True, data={"products": items}, summary=summary)


# ====================== 订单查询 ======================


class OrderQueryArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)

    @field_validator("order_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("-", "").isalnum():
            raise ValueError("订单号只能包含字母、数字和连字符")
        return v


class OrderQueryTool(BaseTool[OrderQueryArgs]):
    name = "order_query"
    description = "查询订单状态、金额与物流单号。只能查询当前登录用户自己的订单"
    args_schema = OrderQueryArgs
    requires_auth = True  # 订单是用户私有数据

    async def _run(self, args: OrderQueryArgs, ctx: ToolContext) -> ToolResult:
        order = ORDERS.get(_norm_id(args.order_id))
        if order is None:
            # 与越权场景返回同样的话术，避免订单号枚举探测
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="order_not_found",
                error_message="未找到该订单，或该订单不属于当前登录账号",
                summary="订单不存在",
            )
        assert_order_ownership(order, ctx.user_id)  # 越权直接抛 ForbiddenError

        data = {
            "order_id": order.order_id,
            "status": order.status.value,
            "product_name": order.product_name,
            "quantity": order.quantity,
            "amount": order.amount,
            "created_at": order.created_at.isoformat(),
            "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
            "tracking_no": order.tracking_no,
        }
        return ToolResult(
            ok=True,
            data=data,
            summary=f"订单 {order.order_id}：{order.product_name} ×{order.quantity}，"
            f"金额 {order.amount} 元，当前状态 {order.status.value}",
        )


# ====================== 物流查询 ======================


class LogisticsQueryArgs(BaseModel):
    order_id: str | None = Field(default=None, max_length=40)
    tracking_no: str | None = Field(default=None, max_length=40)


class LogisticsQueryTool(BaseTool[LogisticsQueryArgs]):
    name = "logistics_query"
    description = "查询订单的物流轨迹。通过订单号查询时会校验订单归属"
    args_schema = LogisticsQueryArgs
    requires_auth = True

    async def _run(self, args: LogisticsQueryArgs, ctx: ToolContext) -> ToolResult:
        tracking_no = args.tracking_no.strip() if args.tracking_no else None

        if args.order_id:
            order = ORDERS.get(_norm_id(args.order_id))
            if order is None:
                return ToolResult(
                    ok=False,
                    status=ToolStatus.FAILED,
                    error_code="order_not_found",
                    error_message="未找到该订单，或该订单不属于当前登录账号",
                    summary="订单不存在",
                )
            assert_order_ownership(order, ctx.user_id)
            if not order.tracking_no:
                return ToolResult(
                    ok=False,
                    status=ToolStatus.FAILED,
                    error_code="not_shipped",
                    error_message="订单尚未发货，暂无物流信息",
                    summary="订单未发货",
                )
            tracking_no = order.tracking_no
        elif tracking_no:
            # 直接用运单号查询时，必须确认该运单属于当前用户的某个订单，否则是越权
            owned = any(o.tracking_no == tracking_no and o.user_id == ctx.user_id for o in ORDERS.values())
            if not owned:
                from ..errors import ForbiddenError

                raise ForbiddenError("未找到该运单，或该运单不属于当前登录账号")
        else:
            raise ToolError("请提供订单号或物流单号")

        logistics = LOGISTICS.get(tracking_no)
        if logistics is None:
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="logistics_not_found",
                error_message="物流信息暂未同步，请稍后再试",
                summary="无物流记录",
            )

        events = [
            {"time": e.time.isoformat(), "location": e.location, "description": e.description} for e in logistics.events
        ]
        latest = logistics.events[-1] if logistics.events else None
        return ToolResult(
            ok=True,
            data={
                "tracking_no": logistics.tracking_no,
                "carrier": logistics.carrier,
                "status": logistics.status,
                "events": events,
            },
            summary=f"{logistics.carrier} {logistics.tracking_no} 当前状态：{logistics.status}"
            + (f"，最新轨迹：{latest.description}（{latest.location}）" if latest else ""),
        )


# ====================== 退货资格预检查 ======================


class ReturnEligibilityArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)


class ReturnEligibilityTool(BaseTool[ReturnEligibilityArgs]):
    name = "return_eligibility_check"
    description = "预检查订单是否满足 7 天无理由退货条件。结论由后端规则判定"
    args_schema = ReturnEligibilityArgs
    requires_auth = True

    async def _run(self, args: ReturnEligibilityArgs, ctx: ToolContext) -> ToolResult:
        order = ORDERS.get(_norm_id(args.order_id))
        if order is None:
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="order_not_found",
                error_message="未找到该订单，或该订单不属于当前登录账号",
                summary="订单不存在",
            )
        assert_order_ownership(order, ctx.user_id)

        product = PRODUCTS.get(order.product_id)
        eligibility = check_return_eligibility(order, product)

        summary = (
            f"订单 {order.order_id} 符合 {eligibility.policy_days} 天无理由退货条件"
            if eligibility.eligible
            else f"订单 {order.order_id} 不符合无理由退货条件：" + "；".join(eligibility.reasons)
        )
        return ToolResult(
            ok=True,  # 判定成功就是 ok，"不符合条件"是业务结论不是失败
            data=eligibility.model_dump(mode="json"),
            summary=summary,
        )


# ====================== 售后申请创建（写操作）======================

#: 进程内幂等存储。生产环境应落库（见 ROADMAP）。
_TICKETS: dict[str, AftersaleTicket] = {}
_IDEMPOTENCY: dict[str, str] = {}


def reset_tickets() -> None:
    """测试用：清空工单与幂等记录。"""
    _TICKETS.clear()
    _IDEMPOTENCY.clear()


class AftersaleCreateArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)
    reason: str = Field(min_length=2, max_length=200)


class AftersaleCreateTool(BaseTool[AftersaleCreateArgs]):
    name = "aftersale_create"
    description = "为符合条件的订单创建售后申请工单（写操作，幂等）"
    args_schema = AftersaleCreateArgs
    requires_auth = True
    is_write = True

    async def _run(self, args: AftersaleCreateArgs, ctx: ToolContext) -> ToolResult:
        order = ORDERS.get(_norm_id(args.order_id))
        if order is None:
            return ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="order_not_found",
                error_message="未找到该订单，或该订单不属于当前登录账号",
                summary="订单不存在",
            )
        assert_order_ownership(order, ctx.user_id)

        # 写操作前必须再跑一遍规则：不能因为模型说"可以退"就创建工单
        eligibility = check_return_eligibility(order, PRODUCTS.get(order.product_id))
        if not eligibility.eligible:
            raise BusinessRuleError("该订单不满足售后申请条件：" + "；".join(eligibility.reasons))

        # 幂等：同一用户 + 同一订单 + 同一原因，只会产生一张工单
        key = hashlib.sha256(f"{ctx.user_id}|{order.order_id}|{args.reason.strip()}".encode()).hexdigest()[:16]
        if existing_id := _IDEMPOTENCY.get(key):
            ticket = _TICKETS[existing_id]
            logger.info("aftersale_idempotent_hit", extra={"ticket_id": ticket.ticket_id})
            raise DuplicateSubmissionError(f"该订单已存在相同原因的售后申请（工单号 {ticket.ticket_id}），请勿重复提交")

        ticket_id = f"AS{utcnow():%Y%m%d}{len(_TICKETS) + 1:04d}"
        ticket = AftersaleTicket(
            ticket_id=ticket_id,
            order_id=order.order_id,
            user_id=ctx.user_id or "",
            reason=args.reason.strip(),
        )
        _TICKETS[ticket_id] = ticket
        _IDEMPOTENCY[key] = ticket_id
        logger.info("aftersale_created", extra={"ticket_id": ticket_id, "order_id": order.order_id})

        return ToolResult(
            ok=True,
            data=ticket.model_dump(mode="json"),
            summary=f"售后申请已提交，工单号 {ticket_id}，客服将在 24 小时内联系您",
        )


# ====================== 高风险操作（拦截示例）======================


class RefundArgs(BaseModel):
    order_id: str = Field(min_length=4, max_length=40)
    amount: float = Field(gt=0)


class RefundTool(BaseTool[RefundArgs]):
    """退款工具。

    **永远不会真正执行**：`is_high_risk=True` 使它在 `BaseTool.execute` 阶段就被拦截，
    除非上下文里有明确的人工审批标记。它存在的意义是展示高风险操作的处理路径——
    模型识别出退款意图 → 框架拦截 → 转人工，而不是让模型自行决定退钱。
    """

    name = "refund_order"
    description = "发起订单退款（高风险操作，需人工审批）"
    args_schema = RefundArgs
    requires_auth = True
    is_write = True
    is_high_risk = True

    async def _run(self, args: RefundArgs, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        # 即使审批通过走到这里，当前版本也不实现真实退款——如实声明而不是假装成功
        raise ToolError("退款能力尚未对接支付系统，请转交人工客服处理")


def build_registry() -> Any:
    from .base import ToolRegistry

    registry = ToolRegistry()
    for tool in (
        ProductQueryTool(),
        OrderQueryTool(),
        LogisticsQueryTool(),
        ReturnEligibilityTool(),
        AftersaleCreateTool(),
        RefundTool(),
    ):
        registry.register(tool)
    return registry
