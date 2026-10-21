"""统一配置。所有密钥走环境变量，代码中不出现任何明文凭据。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "deepseek"]
EmbeddingProvider = Literal["hash", "deepseek_compatible"]
RetrievalMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    """应用配置，环境变量前缀 ``SMA_``。"""

    model_config = SettingsConfigDict(env_prefix="SMA_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "smartmall-agent"
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # ---------- LLM ----------
    llm_provider: LLMProvider = "mock"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ---------- Embedding ----------
    # hash：本地确定性哈希向量，无需任何外部服务（默认，保证离线可测）
    # deepseek_compatible：OpenAI 兼容的 /embeddings 接口
    embedding_provider: EmbeddingProvider = "hash"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = "https://api.deepseek.com/v1"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dim: int = Field(default=256, ge=32, le=4096)
    embedding_timeout_seconds: float = 30.0
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # ---------- 知识库 ----------
    knowledge_dir: Path = Path("data/knowledge")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "smartmall_kb"
    chunk_size: int = Field(default=320, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    # ---------- 检索 ----------
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int = Field(default=10, ge=1, le=50, description="融合前每路召回数量")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF 平滑常数")
    min_score: float = Field(default=0.0, description="低于该分数的候选直接丢弃")
    context_max_chars: int = Field(default=4000, ge=500, le=20000)

    # ---------- Agent ----------
    max_tool_calls: int = Field(default=3, ge=1, le=10, description="单轮对话最多工具调用次数")
    tool_timeout_seconds: float = 10.0
    session_max_turns: int = Field(default=10, ge=1, le=100)

    # ---------- 持久化 ----------
    db_path: Path = Path("data/sma.db")

    # ---------- 安全 ----------
    api_admin_token: SecretStr = SecretStr("")
    reindex_requires_admin: bool = True

    @field_validator("knowledge_dir", "chroma_dir", "db_path", mode="before")
    @classmethod
    def _expand(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()

    @staticmethod
    def _reveal(value: SecretStr | str) -> str:
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    @property
    def llm_api_key_value(self) -> str:
        return self._reveal(self.llm_api_key)

    @property
    def embedding_api_key_value(self) -> str:
        return self._reveal(self.embedding_api_key)

    @property
    def admin_token_value(self) -> str:
        return self._reveal(self.api_admin_token)

    def require_llm_key(self) -> None:
        if self.llm_provider != "mock" and not self.llm_api_key_value:
            raise ValueError(
                f"LLM provider={self.llm_provider} 需要配置 SMA_LLM_API_KEY，"
                "请参考 .env.example 或改用 SMA_LLM_PROVIDER=mock"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
