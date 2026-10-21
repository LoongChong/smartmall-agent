"""API 出入参模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import AnswerStatus, Citation, Intent, RetrievalTrace, ToolCallRecord


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    session_id: str | None = Field(default=None, max_length=64, description="会话 ID，不传则新建")
    include_trace: bool = Field(default=False, description="是否返回检索过程（排查用）")


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    status: AnswerStatus
    intent: Intent
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    needs_human: bool = False
    elapsed_ms: int = 0


class ReindexRequest(BaseModel):
    mode: str = Field(default="incremental", pattern="^(incremental|full)$")


class ReindexResponse(BaseModel):
    ok: bool
    mode: str
    docs_total: int
    docs_indexed: int
    docs_skipped: int
    docs_removed: int
    chunks_written: int
    collection_count: int
    elapsed_ms: int
    failures: list[dict[str, str]] = Field(default_factory=list)


class KnowledgeStatusResponse(BaseModel):
    collection: str
    chunk_count: int
    doc_count: int
    doc_versions: dict[str, str]
    bm25_ready: bool
    bm25_size: int
    embedding_provider: str
    retrieval_mode: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    embedding_provider: str
    knowledge_ready: bool
    chunk_count: int


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: str | None = None
    trace_id: str = ""
