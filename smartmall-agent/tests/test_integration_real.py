"""真实服务集成测试。默认跳过，需显式配置密钥并加 -m integration。

    SMA_LLM_PROVIDER=deepseek SMA_LLM_API_KEY=sk-xxx uv run pytest -m integration

会产生真实费用与网络调用，CI 中不运行。
"""

from __future__ import annotations

import os

import pytest

from sma.agent.service import ChatService
from sma.config import Settings
from sma.llm.base import LLMMessage
from sma.llm.factory import build_embedder, build_llm_client
from sma.models import AnswerStatus
from sma.persistence.session_store import SessionStore
from sma.rag.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration

_HAS_LLM = os.getenv("SMA_LLM_PROVIDER", "mock") != "mock" and bool(os.getenv("SMA_LLM_API_KEY"))
_HAS_EMB = os.getenv("SMA_EMBEDDING_PROVIDER", "hash") != "hash" and bool(os.getenv("SMA_EMBEDDING_API_KEY"))

requires_llm = pytest.mark.skipif(not _HAS_LLM, reason="未配置真实 LLM（SMA_LLM_PROVIDER / SMA_LLM_API_KEY）")
requires_emb = pytest.mark.skipif(
    not _HAS_EMB, reason="未配置真实 Embedding（SMA_EMBEDDING_PROVIDER / SMA_EMBEDDING_API_KEY）"
)


@requires_llm
async def test_real_llm_intent_detection() -> None:
    settings = Settings()
    llm = build_llm_client(settings)
    try:
        resp = await llm.complete([LLMMessage(role="user", content="用一句话介绍 RAG")])
        assert resp.content.strip()
    finally:
        await llm.aclose()


@requires_emb
async def test_real_embedding_dimension() -> None:
    settings = Settings()
    emb = build_embedder(settings)
    try:
        vectors = await emb.embed(["退货政策", "物流查询"])
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1]) > 0
    finally:
        await emb.aclose()


@requires_llm
async def test_real_end_to_end_chat(tmp_path, knowledge_dir) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        knowledge_dir=knowledge_dir,
        chroma_dir=tmp_path / "chroma",
        db_path=tmp_path / "real.db",
        collection_name="real_kb",
    )
    store = await SessionStore.open(settings.db_path)
    service = ChatService(
        settings=settings,
        session_store=store,
        vector_store=ChromaVectorStore(persist_dir=settings.chroma_dir, collection_name=settings.collection_name),
    )
    try:
        await service.reindex(mode="full")
        answer = await service.chat(question="7 天无理由退货从什么时候开始算")
        assert answer.status == AnswerStatus.ANSWERED
        # 真实模型也必须遵守引用约束
        valid_ids = {c.chunk_id for c in answer.citations}
        assert valid_ids, "真实模型的回答也必须带可追溯的引用"
    finally:
        await service.aclose()
        await store.close()


@requires_llm
async def test_real_model_refuses_unknown_question(tmp_path, knowledge_dir) -> None:  # type: ignore[no-untyped-def]
    """真实模型在知识库无相关内容时应拒答而不是编造。"""
    settings = Settings(
        knowledge_dir=knowledge_dir,
        chroma_dir=tmp_path / "chroma2",
        db_path=tmp_path / "real2.db",
        collection_name="real_kb2",
    )
    store = await SessionStore.open(settings.db_path)
    service = ChatService(settings=settings, session_store=store)
    try:
        await service.reindex(mode="full")
        answer = await service.chat(question="你们公司 CEO 的私人手机号是多少")
        assert answer.status in {AnswerStatus.NO_ANSWER, AnswerStatus.ESCALATED}
    finally:
        await service.aclose()
        await store.close()
