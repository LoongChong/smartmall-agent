"""工具：参数校验、权限边界、业务规则、超时、幂等、高风险拦截。"""

from __future__ import annotations

import asyncio

import pytest

from sma.errors import BusinessRuleError, DuplicateSubmissionError, ForbiddenError
from sma.models import OrderStatus, ToolStatus
from sma.tools.base import BaseTool, ToolContext, ToolRegistry, ToolResult
from sma.tools.business_rules import assert_order_ownership, check_return_eligibility
from sma.tools.business_tools import (
    AftersaleCreateTool,
    LogisticsQueryTool,
    OrderQueryTool,
    ProductQueryTool,
    RefundTool,
    ReturnEligibilityTool,
    build_registry,
    reset_tickets,
)
from sma.tools.mock_data import ORDERS, PRODUCTS


@pytest.fixture(autouse=True)
def _clean_tickets():  # type: ignore[no-untyped-def]
    reset_tickets()
    yield
    reset_tickets()


def ctx(user_id: str | None = "U001", *, authenticated: bool = True, approved: bool = False) -> ToolContext:
    return ToolContext(user_id=user_id, session_id="s_test", authenticated=authenticated, approved_high_risk=approved)


# ---------------- 参数校验 ----------------


async def test_invalid_args_rejected_before_execution() -> None:
    result, record = await OrderQueryTool().execute({"order_id": "a"}, ctx())
    assert not result.ok
    assert record.status == ToolStatus.INVALID_ARGS
    assert record.error_code == "tool_validation_error"


async def test_order_id_charset_validated() -> None:
    result, record = await OrderQueryTool().execute({"order_id": "SO2026'; DROP TABLE--"}, ctx())
    assert record.status == ToolStatus.INVALID_ARGS
    assert not result.ok


async def test_missing_required_arg() -> None:
    _, record = await OrderQueryTool().execute({}, ctx())
    assert record.status == ToolStatus.INVALID_ARGS


async def test_product_query_requires_one_of_args() -> None:
    result, record = await ProductQueryTool().execute({}, ctx(None, authenticated=False))
    assert not result.ok
    assert record.status == ToolStatus.FAILED


# ---------------- 权限 ----------------


async def test_order_query_requires_auth() -> None:
    result, record = await OrderQueryTool().execute({"order_id": "SO20261001001"}, ctx(None, authenticated=False))
    assert record.status == ToolStatus.FORBIDDEN
    assert record.error_code == "unauthorized"
    assert not result.ok


async def test_order_ownership_enforced() -> None:
    """U001 查 U002 的订单必须被拒。"""
    result, record = await OrderQueryTool().execute({"order_id": "SO20261005004"}, ctx("U001"))
    assert record.status == ToolStatus.FORBIDDEN
    assert not result.ok
    assert "不属于当前登录账号" in (result.error_message or "")


async def test_own_order_accessible() -> None:
    result, record = await OrderQueryTool().execute({"order_id": "SO20261001001"}, ctx("U001"))
    assert result.ok and record.status == ToolStatus.SUCCESS
    assert result.data["status"] == OrderStatus.DELIVERED.value


async def test_nonexistent_order_message_matches_forbidden() -> None:
    """不存在的订单与越权返回同样话术，防止通过错误信息枚举订单号。"""
    missing, _ = await OrderQueryTool().execute({"order_id": "SO99999999999"}, ctx("U001"))
    forbidden, _ = await OrderQueryTool().execute({"order_id": "SO20261005004"}, ctx("U001"))
    assert missing.error_message == forbidden.error_message


async def test_product_query_is_public() -> None:
    result, record = await ProductQueryTool().execute({"product_id": "P1001"}, ctx(None, authenticated=False))
    assert result.ok and record.status == ToolStatus.SUCCESS


def test_assert_ownership_rejects_anonymous() -> None:
    with pytest.raises(ForbiddenError):
        assert_order_ownership(ORDERS["SO20261001001"], None)


# ---------------- 物流 ----------------


async def test_logistics_by_own_order() -> None:
    result, _ = await LogisticsQueryTool().execute({"order_id": "SO20261001001"}, ctx("U001"))
    assert result.ok
    assert result.data["carrier"] == "顺丰速运"


async def test_logistics_other_users_order_forbidden() -> None:
    _, record = await LogisticsQueryTool().execute({"order_id": "SO20261005004"}, ctx("U001"))
    assert record.status == ToolStatus.FORBIDDEN


async def test_logistics_by_tracking_no_checks_ownership() -> None:
    """知道运单号也不能查别人的：必须验证该运单属于当前用户的订单。"""
    _, record = await LogisticsQueryTool().execute({"tracking_no": "SF1234567890"}, ctx("U002"))
    assert record.status == ToolStatus.FORBIDDEN

    result, _ = await LogisticsQueryTool().execute({"tracking_no": "SF1234567890"}, ctx("U001"))
    assert result.ok


async def test_logistics_unshipped_order() -> None:
    result, _ = await LogisticsQueryTool().execute({"order_id": "SO20261005004"}, ctx("U002"))
    assert not result.ok
    assert result.error_code == "not_shipped"


# ---------------- 退货规则 ----------------


def test_return_eligible_within_window() -> None:
    order = ORDERS["SO20261001001"]  # 签收 3 天
    result = check_return_eligibility(order, PRODUCTS[order.product_id])
    assert result.eligible
    assert result.days_since_delivery == 3
    assert result.deadline is not None


def test_return_rejected_after_window() -> None:
    order = ORDERS["SO20260901002"]  # 签收 20 天
    result = check_return_eligibility(order, PRODUCTS[order.product_id])
    assert not result.eligible
    assert any("超出" in r for r in result.reasons)
    assert result.days_since_delivery == 20


def test_return_rejected_when_not_delivered() -> None:
    order = ORDERS["SO20261010003"]  # 运输中
    result = check_return_eligibility(order, PRODUCTS[order.product_id])
    assert not result.eligible
    assert any("尚未签收" in r for r in result.reasons)


def test_custom_product_not_returnable() -> None:
    """定制商品 return_days=0，即使在时限内也不支持无理由退货。"""
    order = ORDERS["SO20261010003"]
    product = PRODUCTS["P3003"]
    assert product.return_days == 0
    result = check_return_eligibility(order, product)
    assert not result.eligible
    assert any("定制" in r for r in result.reasons)


def test_all_blocking_reasons_collected() -> None:
    """多个原因同时存在时应全部返回，用户一次就能知道所有障碍。"""
    order = ORDERS["SO20261010003"]  # 未签收
    result = check_return_eligibility(order, PRODUCTS["P3003"])  # 且是定制品
    assert len(result.reasons) >= 2


async def test_return_eligibility_tool_reports_business_conclusion() -> None:
    """不符合条件是业务结论，工具本身仍是执行成功（ok=True）。"""
    result, record = await ReturnEligibilityTool().execute({"order_id": "SO20260901002"}, ctx("U001"))
    assert result.ok and record.status == ToolStatus.SUCCESS
    assert result.data["eligible"] is False
    assert "超出" in result.summary


# ---------------- 写操作与幂等 ----------------


async def test_aftersale_create_success() -> None:
    result, record = await AftersaleCreateTool().execute(
        {"order_id": "SO20261001001", "reason": "屏幕有坏点"}, ctx("U001")
    )
    assert result.ok and record.status == ToolStatus.SUCCESS
    assert result.data["ticket_id"].startswith("AS")


async def test_aftersale_duplicate_rejected() -> None:
    args = {"order_id": "SO20261001001", "reason": "屏幕有坏点"}
    first, _ = await AftersaleCreateTool().execute(args, ctx("U001"))
    assert first.ok

    second, record = await AftersaleCreateTool().execute(args, ctx("U001"))
    assert not second.ok
    assert record.error_code == DuplicateSubmissionError("x").code
    assert "请勿重复提交" in (second.error_message or "")


async def test_aftersale_blocked_by_business_rule() -> None:
    """超期订单不能创建售后：即使模型认为可以，代码也会拦住。"""
    result, record = await AftersaleCreateTool().execute({"order_id": "SO20260901002", "reason": "想退"}, ctx("U001"))
    assert not result.ok
    assert record.error_code == BusinessRuleError("x").code
    assert "超出" in (result.error_message or "")


async def test_aftersale_requires_ownership() -> None:
    _, record = await AftersaleCreateTool().execute({"order_id": "SO20261005004", "reason": "想退"}, ctx("U001"))
    assert record.status == ToolStatus.FORBIDDEN


# ---------------- 高风险拦截 ----------------


async def test_refund_tool_always_blocked_without_approval() -> None:
    result, record = await RefundTool().execute({"order_id": "SO20261001001", "amount": 100.0}, ctx("U001"))
    assert not result.ok
    assert record.status == ToolStatus.NEEDS_APPROVAL
    assert record.error_code == "high_risk_requires_approval"


async def test_refund_not_executed_even_when_model_wants_it() -> None:
    """高风险拦截发生在 execute 框架层，_run 根本不会被调用。"""
    called = {"n": 0}

    class SpyRefund(RefundTool):
        async def _run(self, args, ctx):  # type: ignore[no-untyped-def]
            called["n"] += 1
            return ToolResult(ok=True, summary="不该执行到这里")

    await SpyRefund().execute({"order_id": "SO20261001001", "amount": 1.0}, ctx("U001"))
    assert called["n"] == 0


# ---------------- 超时与异常 ----------------


async def test_tool_timeout() -> None:
    from pydantic import BaseModel

    class SlowArgs(BaseModel):
        pass

    class SlowTool(BaseTool[SlowArgs]):
        name = "slow"
        args_schema = SlowArgs

        async def _run(self, args: SlowArgs, c: ToolContext) -> ToolResult:
            await asyncio.sleep(5)
            return ToolResult(ok=True)

    started = asyncio.get_running_loop().time()
    result, record = await SlowTool().execute({}, ctx(), timeout=0.05)
    elapsed = asyncio.get_running_loop().time() - started

    assert record.status == ToolStatus.TIMEOUT
    assert not result.ok
    assert elapsed < 1.0, "超时应立即中断"


async def test_unexpected_exception_does_not_leak() -> None:
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    class BoomTool(BaseTool[Args]):
        name = "boom"
        args_schema = Args

        async def _run(self, args: Args, c: ToolContext) -> ToolResult:
            raise RuntimeError("内部实现细节：数据库连接池耗尽 at line 42")

    result, record = await BoomTool().execute({}, ctx())
    assert record.status == ToolStatus.FAILED
    assert result.summary == "工具执行异常", "面向用户的摘要不应暴露内部细节"


# ---------------- 注册表 ----------------


def test_registry_lists_tools() -> None:
    registry = build_registry()
    assert set(registry.names()) == {
        "product_query",
        "order_query",
        "logistics_query",
        "return_eligibility_check",
        "aftersale_create",
        "refund_order",
    }


def test_registry_rejects_duplicate() -> None:
    registry = ToolRegistry()
    registry.register(ProductQueryTool())
    with pytest.raises(ValueError, match="重复注册"):
        registry.register(ProductQueryTool())


def test_registry_describe_exposes_permissions() -> None:
    """工具的权限属性必须可审计。"""
    described = {d["name"]: d for d in build_registry().describe()}
    assert described["order_query"]["requires_auth"] is True
    assert described["product_query"]["requires_auth"] is False
    assert described["refund_order"]["is_high_risk"] is True
    assert described["aftersale_create"]["is_write"] is True
    assert "args" in described["order_query"]


def test_unknown_tool_returns_none() -> None:
    assert build_registry().get("nonexistent") is None
