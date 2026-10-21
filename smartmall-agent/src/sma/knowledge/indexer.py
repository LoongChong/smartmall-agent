"""知识库构建：加载 → 切块 → 嵌入 → 写入 Chroma → 重建 BM25。

更新策略（三种）：
- ``incremental``（默认）：比对每篇文档的 version（内容哈希），只重建变化的文档。
  重复执行不产生任何写入 —— 这是"重复构建"测试的核心断言。
- ``full``：删除整个 collection 后全量重建。用于切换 embedding 模型等破坏性变更。
- 删除：文档在磁盘上被移除时，索引里的对应块也会被清理。

失败诊断：每篇文档的处理结果（新增/更新/跳过/失败）都记录在 ``IndexReport`` 里，
不是只返回一个总数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import Settings
from ..errors import IndexingError
from ..logging_setup import get_logger
from ..models import Chunk
from ..rag.bm25 import BM25Index
from ..rag.embedding import Embedder
from ..rag.vector_store import ChromaVectorStore
from .chunker import chunk_document
from .loader import load_directory

logger = get_logger(__name__)

IndexMode = Literal["incremental", "full"]


@dataclass
class IndexReport:
    mode: str = "incremental"
    docs_total: int = 0
    docs_indexed: int = 0
    docs_skipped: int = 0
    docs_removed: int = 0
    chunks_written: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    elapsed_ms: int = 0
    collection_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "docs_total": self.docs_total,
            "docs_indexed": self.docs_indexed,
            "docs_skipped": self.docs_skipped,
            "docs_removed": self.docs_removed,
            "chunks_written": self.chunks_written,
            "failures": self.failures,
            "elapsed_ms": self.elapsed_ms,
            "collection_count": self.collection_count,
            "ok": self.ok,
        }


class KnowledgeIndexer:
    def __init__(self, *, settings: Settings, store: ChromaVectorStore, embedder: Embedder) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder

    async def build(self, *, mode: IndexMode = "incremental", directory: Path | None = None) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport(mode=mode)
        directory = directory or self._settings.knowledge_dir

        if mode == "full":
            self._store.reset()

        docs = load_directory(directory)
        report.docs_total = len(docs)
        existing = self._store.doc_versions() if mode == "incremental" else {}

        for meta, body in docs:
            try:
                if mode == "incremental" and existing.get(meta.doc_id) == meta.version:
                    report.docs_skipped += 1
                    continue

                chunks = chunk_document(
                    meta,
                    body,
                    chunk_size=self._settings.chunk_size,
                    overlap=self._settings.chunk_overlap,
                )
                if not chunks:
                    raise IndexingError(f"文档切块结果为空: {meta.doc_id}")

                # 内容变化时先删旧块：新版本块数可能更少，仅 upsert 会残留孤儿块
                if meta.doc_id in existing:
                    self._store.delete_doc(meta.doc_id)

                embeddings = await self._embedder.embed([c.content for c in chunks])
                self._store.upsert(chunks, embeddings)
                report.docs_indexed += 1
                report.chunks_written += len(chunks)
            except Exception as exc:  # noqa: BLE001 - 单篇失败不应中断整体构建
                logger.exception("index_doc_failed", extra={"doc_id": meta.doc_id})
                report.failures.append({"doc_id": meta.doc_id, "error": str(exc)[:300]})

        # 清理磁盘上已删除的文档
        if mode == "incremental":
            on_disk = {meta.doc_id for meta, _ in docs}
            for doc_id in existing:
                if doc_id not in on_disk:
                    self._store.delete_doc(doc_id)
                    report.docs_removed += 1

        report.collection_count = self._store.count()
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("index_done", extra=report.to_dict())
        return report

    def build_bm25(self) -> BM25Index:
        """从向量库里取全量块重建 BM25。两个索引共用同一份语料，避免不一致。"""
        chunks: list[Chunk] = self._store.all_chunks()
        return BM25Index(chunks)
