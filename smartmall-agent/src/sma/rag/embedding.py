"""Embedding 抽象与两种实现。

``HashEmbedding``（默认）：本地确定性哈希向量，无需任何外部服务。
它**不是真正的语义向量**——这一点必须说清楚，文档里也写明了。
它的作用是让向量检索链路（索引、相似度、融合、评测）在离线环境下可完整测试，
而不是假装具备语义理解能力。同义词检索能力主要由 BM25 分词和混合检索承担。

``ApiEmbedding``：OpenAI 兼容的 /embeddings 接口，配置真实 Key 后替换上去，
索引与检索代码零改动。
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import httpx

from ..errors import EmbeddingError, LLMRateLimitError, LLMTimeoutError
from ..logging_setup import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


class Embedder(ABC):
    name: str = "base"
    dim: int = 256

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def aclose(self) -> None:  # pragma: no cover
        return None


class HashEmbedding(Embedder):
    """字符/词级哈希 + TF 加权的稀疏向量投影。

    算法：把文本切成中文单字与英文词，每个 token 用 sha256 映射到固定维度的下标，
    按 TF 累加后做 L2 归一化。相同词汇重叠度高的文本余弦相似度高——
    本质是**词面相似**，不是语义相似。
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        if dim < 32:
            raise EmbeddingError("embedding 维度过小，至少 32")
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class ApiEmbedding(Embedder):
    """OpenAI 兼容的 embeddings 接口。"""

    name = "api"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int = 1536,
        timeout: float = 30.0,
        batch_size: int = 32,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("API Embedding 需要配置 SMA_EMBEDDING_API_KEY")
        self.dim = dim
        self._model = model
        self._batch_size = batch_size
        self._endpoint = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(8.0, timeout)))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(await self._embed_batch(texts[i : i + self._batch_size]))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                self._endpoint, json={"model": self._model, "input": batch}, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Embedding 调用超时", detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMRateLimitError("Embedding 网络错误", detail=str(exc)) from exc

        if resp.status_code == 429:
            raise LLMRateLimitError("Embedding 限流 (HTTP 429)")
        if resp.status_code >= 500:
            raise LLMRateLimitError(f"Embedding 服务端错误 (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            raise EmbeddingError(f"Embedding 请求被拒绝 (HTTP {resp.status_code})", detail=resp.text[:300])

        try:
            data = resp.json()["data"]
            vectors = [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("Embedding 响应结构异常", detail=str(exc)) from exc

        if len(vectors) != len(batch):
            raise EmbeddingError(f"Embedding 返回数量不匹配: 期望 {len(batch)}，实际 {len(vectors)}")
        return vectors

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise EmbeddingError(f"向量维度不一致: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
