"""测试夹具。全部离线运行：hash embedding + mock LLM + 临时 Chroma 目录。"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from sma.agent.service import ChatService
from sma.config import Settings
from sma.knowledge.indexer import KnowledgeIndexer
from sma.llm.mock import MockLLMClient
from sma.logging_setup import setup_logging
from sma.persistence.session_store import SessionStore
from sma.rag.embedding import HashEmbedding
from sma.rag.vector_store import ChromaVectorStore
from sma.tools.business_tools import build_registry, reset_tickets

setup_logging("ERROR", json_output=False)

KNOWLEDGE_SRC = Path(__file__).resolve().parents[1] / "data" / "knowledge"


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """把真实知识库复制到临时目录，避免测试污染源文件。"""
    target = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_SRC, target)
    return target


@pytest.fixture
def settings(tmp_path: Path, knowledge_dir: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_provider="hash",
        embedding_dim=256,
        knowledge_dir=knowledge_dir,
        chroma_dir=tmp_path / "chroma",
        db_path=tmp_path / "sma.db",
        collection_name="test_kb",
        chunk_size=320,
        chunk_overlap=64,
        top_k=4,
        candidate_k=10,
        tool_timeout_seconds=2.0,
        session_max_turns=5,
    )


@pytest.fixture
def embedder(settings: Settings) -> HashEmbedding:
    return HashEmbedding(dim=settings.embedding_dim)


@pytest.fixture
def vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=settings.chroma_dir, collection_name=settings.collection_name)


@pytest.fixture
def indexer(settings: Settings, vector_store: ChromaVectorStore, embedder: HashEmbedding) -> KnowledgeIndexer:
    return KnowledgeIndexer(settings=settings, store=vector_store, embedder=embedder)


@pytest_asyncio.fixture
async def session_store(settings: Settings) -> AsyncIterator[SessionStore]:
    store = await SessionStore.open(settings.db_path, max_turns=settings.session_max_turns)
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def llm() -> MockLLMClient:
    return MockLLMClient()


@pytest_asyncio.fixture
async def service(
    settings: Settings,
    session_store: SessionStore,
    llm: MockLLMClient,
    embedder: HashEmbedding,
    vector_store: ChromaVectorStore,
) -> AsyncIterator[ChatService]:
    reset_tickets()
    svc = ChatService(
        settings=settings,
        session_store=session_store,
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        registry=build_registry(),
    )
    await svc.reindex(mode="incremental")
    try:
        yield svc
    finally:
        await svc.aclose()
        reset_tickets()
