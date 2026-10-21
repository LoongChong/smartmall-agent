"""DeepSeek（OpenAI 兼容 Chat Completions）客户端。

只依赖 httpx，不引入厂商 SDK：接口是 OpenAI 兼容的，换成任意兼容网关只需改 base_url。
超时、重试、错误分类都在这里收口。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ..errors import LLMError, LLMRateLimitError, LLMTimeoutError
from ..logging_setup import get_logger
from .base import LLMClient, LLMMessage, LLMResponse

logger = get_logger(__name__)


class DeepSeekClient(LLMClient):
    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("缺少 LLM API Key")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # 统一拼绝对 URL：注入自定义 client（测试 / 自带连接池）时也遵守同一个 base_url
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
        )
        # 鉴权头随每次请求下发，而不是绑在 client 上：
        # 否则调用方注入自己的 AsyncClient 时会静默丢失鉴权。
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "stream": False,
        }
        if response_format := kwargs.pop("response_format", None):
            payload["response_format"] = response_format

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(self._endpoint, json=payload, headers=self._headers)
                if resp.status_code == 429:
                    raise LLMRateLimitError("LLM 限流 (HTTP 429)")
                if resp.status_code >= 500:
                    raise LLMRateLimitError(f"LLM 服务端错误 (HTTP {resp.status_code})")
                if resp.status_code >= 400:
                    raise LLMError(f"LLM 请求被拒绝 (HTTP {resp.status_code})", detail=resp.text[:500])
                data = resp.json()
                choice = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=choice or "",
                    model=data.get("model", self.model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(f"LLM 调用超时：{exc}")
            except httpx.HTTPError as exc:
                last_error = LLMRateLimitError(f"LLM 网络错误：{exc}")
            except (KeyError, IndexError, ValueError) as exc:
                raise LLMError("LLM 响应结构异常", detail=str(exc)) from exc
            except LLMRateLimitError as exc:
                last_error = exc

            if attempt < self.max_retries:
                backoff = min(2**attempt * 0.5, 8.0)
                logger.warning(
                    "llm_retry", extra={"attempt": attempt + 1, "backoff_s": backoff, "err": str(last_error)}
                )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()
