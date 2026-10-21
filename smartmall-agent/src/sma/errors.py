"""统一错误模型。

工具相关的错误单独分类，因为 Agent 需要根据错误类型决定"重试 / 换工具 / 如实告知用户"。
"""

from __future__ import annotations


class SMAError(Exception):
    code: str = "sma_error"
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class RetryableError(SMAError):
    code = "retryable_error"
    http_status = 503


class PermanentError(SMAError):
    code = "permanent_error"


class UserError(SMAError):
    code = "user_error"
    http_status = 400


# ---------- 配置 / 模型 ----------


class ConfigError(PermanentError):
    code = "config_error"


class LLMError(PermanentError):
    code = "llm_error"


class LLMTimeoutError(RetryableError):
    code = "llm_timeout"


class LLMRateLimitError(RetryableError):
    code = "llm_rate_limit"


class LLMOutputFormatError(PermanentError):
    code = "llm_output_format_error"


class EmbeddingError(PermanentError):
    code = "embedding_error"


# ---------- 知识库 / 检索 ----------


class KnowledgeBaseError(PermanentError):
    code = "knowledge_base_error"


class KnowledgeBaseEmptyError(PermanentError):
    """索引为空：必须明确报错，而不是返回"没找到答案"掩盖配置问题。"""

    code = "knowledge_base_empty"
    http_status = 503


class IndexingError(PermanentError):
    code = "indexing_error"


# ---------- 工具 ----------


class ToolError(PermanentError):
    """工具执行失败（业务语义上的失败，如订单不存在）。"""

    code = "tool_error"


class ToolValidationError(UserError):
    """工具参数不合法。"""

    code = "tool_validation_error"


class ToolTimeoutError(RetryableError):
    code = "tool_timeout"


class ToolNotFoundError(PermanentError):
    code = "tool_not_found"


# ---------- 权限 / 业务规则 ----------


class UnauthorizedError(UserError):
    code = "unauthorized"
    http_status = 401


class ForbiddenError(UserError):
    """越权：身份合法但无权访问该资源（如查他人订单）。"""

    code = "forbidden"
    http_status = 403


class BusinessRuleError(UserError):
    """业务规则不满足（如超出退货时限）。属于正常业务结论，不是系统故障。"""

    code = "business_rule_violation"
    http_status = 409


class HighRiskOperationError(UserError):
    """高风险操作需要额外确认或人工审批，模型不得自行决定执行。"""

    code = "high_risk_requires_approval"
    http_status = 403


class DuplicateSubmissionError(UserError):
    code = "duplicate_submission"
    http_status = 409


class SessionNotFoundError(UserError):
    code = "session_not_found"
    http_status = 404
