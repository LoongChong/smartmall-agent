"""真实 HTTP 适配器的错误分类（MockTransport，不触网）。"""

from __future__ import annotations

import httpx
import pytest

from sma.errors import EmbeddingError, LLMError, LLMRateLimitError, LLMTimeoutError
from sma.llm.base import LLMMessage, extract_json
from sma.llm.deepseek import DeepSeekClient
from sma.rag.embedding import ApiEmbedding, cosine


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "deepseek-chat",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        },
    )


# ---------------- JSON 提取 ----------------


def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose() -> None:
    assert extract_json('好的，结果如下：\n{"a": 1}\n希望有帮助') == {"a": 1}


def test_extract_json_failure() -> None:
    from sma.errors import LLMOutputFormatError

    with pytest.raises(LLMOutputFormatError):
        extract_json("完全不是 JSON")


# ---------------- DeepSeek ----------------


async def test_deepseek_success() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: chat_response("你好")))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=0)
    try:
        resp = await llm.complete([LLMMessage(role="user", content="hi")])
        assert resp.content == "你好"
        assert resp.prompt_tokens == 12
    finally:
        await client.aclose()


async def test_deepseek_payload_and_auth() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        captured["auth"] = request.headers.get("authorization")
        return chat_response("{}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = DeepSeekClient(api_key="sk-test", client=client, model="deepseek-chat")
    try:
        await llm.complete([LLMMessage(role="user", content="hi")])
        assert captured["model"] == "deepseek-chat"
        assert captured["stream"] is False
        assert captured["auth"] == "Bearer sk-test"
    finally:
        await client.aclose()


async def test_deepseek_requires_key() -> None:
    with pytest.raises(LLMError):
        DeepSeekClient(api_key="")


async def test_deepseek_429_retries_then_fails() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=1)
    try:
        with pytest.raises(LLMRateLimitError):
            await llm.complete([LLMMessage(role="user", content="hi")])
        assert calls["n"] == 2
    finally:
        await client.aclose()


async def test_deepseek_401_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid key")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = DeepSeekClient(api_key="sk-bad", client=client, max_retries=2)
    try:
        with pytest.raises(LLMError):
            await llm.complete([LLMMessage(role="user", content="hi")])
        assert calls["n"] == 1, "鉴权失败不应重试"
    finally:
        await client.aclose()


async def test_deepseek_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=0)
    try:
        with pytest.raises(LLMTimeoutError):
            await llm.complete([LLMMessage(role="user", content="hi")])
    finally:
        await client.aclose()


async def test_deepseek_malformed_response() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": True})))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=0)
    try:
        with pytest.raises(LLMError):
            await llm.complete([LLMMessage(role="user", content="hi")])
    finally:
        await client.aclose()


async def test_deepseek_complete_json_validates() -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        intent: str

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: chat_response('{"intent": "order_query"}')))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=0)
    try:
        out = await llm.complete_json([LLMMessage(role="user", content="hi")], Out)
        assert out.intent == "order_query"
    finally:
        await client.aclose()


async def test_deepseek_schema_mismatch_raises() -> None:
    from pydantic import BaseModel

    from sma.errors import LLMOutputFormatError

    class Out(BaseModel):
        required_field: int

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: chat_response('{"other": 1}')))
    llm = DeepSeekClient(api_key="sk-test", client=client, max_retries=0)
    try:
        with pytest.raises(LLMOutputFormatError):
            await llm.complete_json([LLMMessage(role="user", content="hi")], Out)
    finally:
        await client.aclose()


# ---------------- API Embedding ----------------


def embedding_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]})


async def test_api_embedding_success() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: embedding_response([[0.1, 0.2], [0.3, 0.4]])))
    emb = ApiEmbedding(api_key="k", base_url="https://x.com/v1", model="m", dim=2, client=client, batch_size=8)
    try:
        vectors = await emb.embed(["a", "b"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    finally:
        await client.aclose()


async def test_api_embedding_batches() -> None:
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        inputs = json.loads(request.content)["input"]
        batches.append(len(inputs))
        return embedding_response([[1.0] for _ in inputs])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    emb = ApiEmbedding(api_key="k", base_url="https://x.com/v1", model="m", dim=1, client=client, batch_size=2)
    try:
        await emb.embed(["a", "b", "c", "d", "e"])
        assert batches == [2, 2, 1]
    finally:
        await client.aclose()


async def test_api_embedding_count_mismatch_detected() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: embedding_response([[1.0]]))  # 只返回 1 条
    )
    emb = ApiEmbedding(api_key="k", base_url="https://x.com/v1", model="m", dim=1, client=client)
    try:
        with pytest.raises(EmbeddingError, match="数量不匹配"):
            await emb.embed(["a", "b"])
    finally:
        await client.aclose()


async def test_api_embedding_rate_limit() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    emb = ApiEmbedding(api_key="k", base_url="https://x.com/v1", model="m", dim=1, client=client)
    try:
        with pytest.raises(LLMRateLimitError):
            await emb.embed(["a"])
    finally:
        await client.aclose()


async def test_api_embedding_4xx_permanent() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")))
    emb = ApiEmbedding(api_key="k", base_url="https://x.com/v1", model="m", dim=1, client=client)
    try:
        with pytest.raises(EmbeddingError):
            await emb.embed(["a"])
    finally:
        await client.aclose()


async def test_api_embedding_requires_key() -> None:
    with pytest.raises(EmbeddingError):
        ApiEmbedding(api_key="", base_url="https://x.com/v1", model="m")


def test_cosine_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingError, match="维度不一致"):
        cosine([1.0], [1.0, 2.0])


def test_cosine_zero_vector() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
