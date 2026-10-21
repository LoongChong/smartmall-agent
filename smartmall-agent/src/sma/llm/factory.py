"""按配置构造 LLM 客户端与 Embedder。"""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from ..rag.embedding import ApiEmbedding, Embedder, HashEmbedding
from .base import LLMClient
from .mock import MockLLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient()
    if settings.llm_provider == "deepseek":
        settings.require_llm_key()
        from .deepseek import DeepSeekClient

        return DeepSeekClient(
            api_key=settings.llm_api_key_value,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    raise ConfigError(f"未知的 LLM provider: {settings.llm_provider}")


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedding(dim=settings.embedding_dim)
    if settings.embedding_provider == "deepseek_compatible":
        if not settings.embedding_api_key_value:
            raise ConfigError(
                "embedding_provider=deepseek_compatible 需要配置 SMA_EMBEDDING_API_KEY，"
                "或改用 SMA_EMBEDDING_PROVIDER=hash"
            )
        return ApiEmbedding(
            api_key=settings.embedding_api_key_value,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            timeout=settings.embedding_timeout_seconds,
            batch_size=settings.embedding_batch_size,
        )
    raise ConfigError(f"未知的 embedding provider: {settings.embedding_provider}")
