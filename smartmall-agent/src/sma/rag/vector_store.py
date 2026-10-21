"""Chroma 向量库封装。

两个关键设计：

1. **索引幂等**：chunk_id 由 doc_id + 序号派生，用 ``upsert`` 写入。
   同一文档重复构建不会产生重复向量；文档内容变化时 version 变化，
   通过"先删该 doc 的全部旧块再写新块"保证不残留脏数据。
2. **嵌入在外部计算**：Chroma 支持自带 embedding function，但我们在外面算好再传入，
   这样 Embedder 可以被独立替换和测试，也避免 Chroma 默认下载 ONNX 模型（离线环境会失败）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import KnowledgeBaseError
from ..logging_setup import get_logger
from ..models import Chunk, DocCategory

logger = get_logger(__name__)


class ChromaVectorStore:
    def __init__(self, *, persist_dir: Path, collection_name: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise KnowledgeBaseError("未安装 chromadb，请执行 uv pip install -e '.[dev]'") from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---------- 写 ----------

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise KnowledgeBaseError(f"块数量与向量数量不一致: {len(chunks)} vs {len(embeddings)}")
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        logger.info("vector_upsert", extra={"count": len(chunks)})
        return len(chunks)

    def delete_doc(self, doc_id: str) -> None:
        """删除一篇文档的全部块。用于内容变化时清理旧块（避免切块数量变少时残留）。"""
        self._collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        """全量重建：删掉整个 collection 再创建。"""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.warning("vector_store_reset", extra={"collection": self._collection_name})

    # ---------- 读 ----------

    def count(self) -> int:
        return self._collection.count()

    def doc_versions(self) -> dict[str, str]:
        """返回 {doc_id: version}，用于增量更新时判断哪些文档需要重建。"""
        got = self._collection.get(include=["metadatas"])
        versions: dict[str, str] = {}
        for meta in got.get("metadatas") or []:
            if meta and (doc_id := meta.get("doc_id")):
                versions[str(doc_id)] = str(meta.get("version", ""))
        return versions

    def query(
        self, embedding: list[float], *, top_k: int, category: DocCategory | None = None
    ) -> list[tuple[Chunk, float]]:
        """返回 (块, 相似度)，相似度 = 1 - cosine 距离，越大越相关。"""
        where: dict[str, Any] | None = {"category": category.value} if category else None
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        out: list[tuple[Chunk, float]] = []
        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists, strict=True):
            out.append((_to_chunk(chunk_id, content, meta or {}), 1.0 - float(dist)))
        return out

    def all_chunks(self) -> list[Chunk]:
        """取出全部块。BM25 索引在内存中构建，需要全量语料。"""
        got = self._collection.get(include=["documents", "metadatas"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        return [_to_chunk(cid, content, meta or {}) for cid, content, meta in zip(ids, docs, metas, strict=True)]


def _to_chunk(chunk_id: str, content: str, meta: dict[str, Any]) -> Chunk:
    try:
        category = DocCategory(str(meta.get("category", DocCategory.GENERAL_FAQ.value)))
    except ValueError:
        category = DocCategory.GENERAL_FAQ
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta.get("doc_id", "")),
        content=content,
        seq=int(meta.get("seq", 0)),
        title=str(meta.get("title", "")),
        category=category,
        product_id=str(meta["product_id"]) if meta.get("product_id") else None,
        version=str(meta.get("version", "")),
    )
