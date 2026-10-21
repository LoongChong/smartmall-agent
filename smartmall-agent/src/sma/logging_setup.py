"""结构化日志 + 请求/会话关联 ID。

用标准库 logging 实现 JSON Formatter，避免为"展示技术栈"引入 structlog。
``trace_id`` 与 ``session_id`` 通过 contextvars 透传，一次对话的全部日志可被串联。
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_session_id: ContextVar[str] = ContextVar("session_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime", "taskName"}


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def set_trace_id(value: str | None = None) -> str:
    tid = value or new_trace_id()
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


def set_session_id(value: str) -> None:
    _session_id.set(value)


def get_session_id() -> str:
    return _session_id.get()


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        record.session_id = _session_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = _safe(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # default=repr：遇到 SecretStr 等不可序列化对象输出其 repr（SecretStr 的 repr 已掩码）
        return json.dumps(payload, ensure_ascii=False, default=repr)


def _safe(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None) | list | dict):
        return value
    return repr(value)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """幂等地初始化根 logger。重复调用只会替换 handler，不会叠加。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # 日志走 stderr：stdout 留给 CLI 的结构化输出（JSON），避免两者混在一起
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s/%(session_id)s] %(name)s: %(message)s")
        )
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel("WARNING")
    # jieba 在 DEBUG 级别会打印词典加载过程，噪声大且无价值
    logging.getLogger("jieba").setLevel("WARNING")
    logging.getLogger("chromadb").setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
