"""工具框架：Schema 校验、权限边界、超时、幂等、执行记录。

核心原则（见 docs/TOOLS_AND_BUSINESS_RULES.md）：

**LLM 只负责识别意图和抽取参数；是否允许执行、业务结论是什么，全部由后端代码决定。**

所以每个工具都显式声明：
- ``args_schema``：Pydantic 模型，参数校验在进入业务逻辑前完成；
- ``requires_auth``：是否需要已认证的用户身份；
- ``is_write``：是否为写操作（写操作走幂等检查）；
- ``is_high_risk``：是否为高风险操作（模型不得自行决定执行，必须走人工/额外确认）。
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import (
    ForbiddenError,
    HighRiskOperationError,
    SMAError,
    ToolTimeoutError,
    ToolValidationError,
    UnauthorizedError,
)
from ..logging_setup import get_logger
from ..models import ToolCallRecord, ToolStatus

logger = get_logger(__name__)

ArgsT = TypeVar("ArgsT", bound=BaseModel)


@dataclass(slots=True)
class ToolContext:
    """调用上下文。

    ``user_id`` 来自**已认证的会话**，不是用户在聊天里自述的身份——
    这是防越权的根本前提（见 SECURITY.md）。
    """

    user_id: str | None
    session_id: str
    authenticated: bool = False
    approved_high_risk: bool = False


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any] | None = None
    summary: str = ""
    error_code: str | None = None
    error_message: str | None = None
    status: ToolStatus = ToolStatus.SUCCESS


class BaseTool(ABC, Generic[ArgsT]):
    name: str = "base"
    description: str = ""
    args_schema: type[BaseModel]
    requires_auth: bool = False
    is_write: bool = False
    is_high_risk: bool = False

    @abstractmethod
    async def _run(self, args: ArgsT, ctx: ToolContext) -> ToolResult: ...

    async def execute(
        self,
        raw_args: dict[str, Any],
        ctx: ToolContext,
        *,
        # ASYNC109 建议用 asyncio.timeout 替代 timeout 参数；这里 timeout 是工具的业务配置项
        # （由 settings.tool_timeout_seconds 注入），内部确实是用 asyncio.timeout 实现的。
        timeout: float = 10.0,  # noqa: ASYNC109
    ) -> tuple[ToolResult, ToolCallRecord]:
        """统一入口：校验 → 鉴权 → 高风险拦截 → 带超时执行 → 产出记录。"""
        started = time.perf_counter()

        def record(result: ToolResult) -> ToolCallRecord:
            return ToolCallRecord(
                tool=self.name,
                status=result.status,
                args={k: str(v)[:100] for k, v in (raw_args or {}).items()},
                error_code=result.error_code,
                error_message=result.error_message,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                result_summary=result.summary[:300],
            )

        # 1. 参数校验
        try:
            args = self.args_schema.model_validate(raw_args or {})
        except ValidationError as exc:
            err = ToolValidationError(f"工具 {self.name} 参数不合法", detail=str(exc)[:400])
            logger.warning("tool_invalid_args", extra={"tool": self.name, "error": err.detail})
            result = ToolResult(
                ok=False,
                status=ToolStatus.INVALID_ARGS,
                error_code=err.code,
                error_message=err.message,
                summary="参数不完整或格式错误",
            )
            return result, record(result)

        # 2. 鉴权：身份必须来自会话
        if self.requires_auth and not (ctx.authenticated and ctx.user_id):
            err = UnauthorizedError(f"工具 {self.name} 需要登录后才能使用")
            result = ToolResult(
                ok=False,
                status=ToolStatus.FORBIDDEN,
                error_code=err.code,
                error_message=err.message,
                summary="需要身份认证",
            )
            logger.warning("tool_unauthorized", extra={"tool": self.name, "session": ctx.session_id})
            return result, record(result)

        # 3. 高风险操作拦截：模型不得自行决定执行
        if self.is_high_risk and not ctx.approved_high_risk:
            err = HighRiskOperationError(f"操作 {self.name} 属于高风险，需要人工审批或二次确认")
            result = ToolResult(
                ok=False,
                status=ToolStatus.NEEDS_APPROVAL,
                error_code=err.code,
                error_message=err.message,
                summary="高风险操作已拦截，转人工处理",
            )
            logger.warning("tool_high_risk_blocked", extra={"tool": self.name, "session": ctx.session_id})
            return result, record(result)

        # 4. 执行
        try:
            async with asyncio.timeout(timeout):
                result = await self._run(args, ctx)  # type: ignore[arg-type]
        except TimeoutError:
            err = ToolTimeoutError(f"工具 {self.name} 执行超时（{timeout}s）")
            logger.warning("tool_timeout", extra={"tool": self.name, "timeout_s": timeout})
            result = ToolResult(
                ok=False,
                status=ToolStatus.TIMEOUT,
                error_code=err.code,
                error_message=err.message,
                summary="查询超时，请稍后重试",
            )
        except ForbiddenError as exc:
            logger.warning("tool_forbidden", extra={"tool": self.name, "error": exc.message})
            result = ToolResult(
                ok=False,
                status=ToolStatus.FORBIDDEN,
                error_code=exc.code,
                error_message=exc.message,
                summary="无权访问该资源",
            )
        except SMAError as exc:
            logger.warning("tool_failed", extra={"tool": self.name, "code": exc.code, "error": exc.message})
            result = ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code=exc.code,
                error_message=exc.message,
                summary=exc.message[:200],
            )
        except Exception as exc:  # noqa: BLE001 - 工具异常不得击穿到用户
            logger.exception("tool_unexpected_error", extra={"tool": self.name})
            result = ToolResult(
                ok=False,
                status=ToolStatus.FAILED,
                error_code="internal_error",
                error_message=str(exc)[:300],
                summary="工具执行异常",
            )

        logger.info(
            "tool_executed",
            extra={"tool": self.name, "status": result.status.value, "ok": result.ok},
        )
        return result, record(result)


class ToolRegistry:
    """工具注册表。Agent 只能调用注册过的工具——这是能力边界的实现。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}

    def register(self, tool: BaseTool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"重复注册工具: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool[Any] | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "requires_auth": t.requires_auth,
                "is_write": t.is_write,
                "is_high_risk": t.is_high_risk,
                "args": t.args_schema.model_json_schema(),
            }
            for t in (self._tools[n] for n in self.names())
        ]
