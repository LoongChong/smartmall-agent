"""检索器：Vector / BM25 / Hybrid(RRF) / Hybrid+Rerank。

融合策略用 **RRF（Reciprocal Rank Fusion）**：

    score(d) = Σ_r 1 / (k + rank_r(d))

为什么不用分数加权相加：向量相似度（0–1 的余弦）和 BM25 分数（无上界的实数）
量纲完全不同，线性加权需要先做归一化，而归一化方式（min-max / z-score）
会随 query 变化而改变相对顺序，不稳定。RRF 只用**排名**，天然跨量纲可比，
而且对单路的异常高分不敏感。k=60 是 RRF 原论文的常用取值。

Reranker 是可选的**最后一步**，只对融合后的候选做重排，不参与召回。
当前提供的是基于词面覆盖率与位置的轻量重排器（`LexicalReranker`），
**不是** cross-encoder 模型——这一点在文档里明确声明，不冒充语义重排。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Settings
from ..errors import KnowledgeBaseEmptyError
from ..logging_setup import get_logger
from ..models import Chunk, DocCategory, RetrievalTrace, RetrievedChunk
from .bm25 import BM25Index, tokenize
from .embedding import Embedder
from .vector_store import ChromaVectorStore

logger = get_logger(__name__)


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    async def retrieve(
        self, query: str, *, top_k: int, category: DocCategory | None = None
    ) -> list[RetrievedChunk]: ...


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, store: ChromaVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def retrieve(self, query: str, *, top_k: int, category: DocCategory | None = None) -> list[RetrievedChunk]:
        if self._store.count() == 0:
            raise KnowledgeBaseEmptyError("向量索引为空，请先执行知识库构建")
        vector = await self._embedder.embed_one(query)
        hits = self._store.query(vector, top_k=top_k, category=category)
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1, retriever=self.name)
            for i, (chunk, score) in enumerate(hits)
        ]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index: BM25Index) -> None:
        self._index = index

    async def retrieve(self, query: str, *, top_k: int, category: DocCategory | None = None) -> list[RetrievedChunk]:
        if self._index.size == 0:
            raise KnowledgeBaseEmptyError("BM25 索引为空，请先执行知识库构建")
        hits = self._index.search(query, top_k=top_k * (3 if category else 1))
        hits = [(c, s) for c, s in hits if c.category == category][:top_k] if category else hits[:top_k]
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1, retriever=self.name)
            for i, (chunk, score) in enumerate(hits)
        ]


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], *, k: int = 60) -> list[RetrievedChunk]:
    """对多路召回做 RRF 融合。返回按融合分降序排列的去重结果。"""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    sources: dict[str, list[str]] = {}

    for ranking in rankings:
        for item in ranking:
            cid = item.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item.rank)
            sources.setdefault(cid, []).append(item.retriever)
            # 保留排名更靠前的那条作为代表（用于展示原始分数）
            if cid not in best or item.rank < best[cid].rank:
                best[cid] = item

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(ordered, 1):
        item = best[cid]
        out.append(
            RetrievedChunk(
                chunk=item.chunk,
                score=round(score, 6),
                rank=rank,
                retriever="+".join(sorted(set(sources[cid]))),
            )
        )
    return out


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]: ...


class LexicalReranker(Reranker):
    """轻量重排：查询词覆盖率 + 首次命中位置 + 标题命中加权。

    **这不是 cross-encoder**。它解决的是一个具体问题：
    RRF 只看排名，当两个块的排名接近时，无法区分"查询词全命中"和"只命中一个词"。
    覆盖率能把前者提上来。代价是对同义改写无能为力（那需要真正的语义模型）。
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return candidates[:top_k]

        scored: list[tuple[float, RetrievedChunk]] = []
        for item in candidates:
            content = item.chunk.content
            content_terms = set(tokenize(content))
            coverage = len(query_terms & content_terms) / len(query_terms)

            # 首次命中位置越靠前越好（答案通常在块开头）
            positions = [content.find(t) for t in query_terms if t in content]
            position_bonus = 0.0
            if positions:
                position_bonus = max(0.0, 1.0 - min(positions) / max(len(content), 1)) * 0.2

            title_terms = set(tokenize(item.chunk.title))
            title_bonus = 0.15 if query_terms & title_terms else 0.0

            # 保留原融合排名的影响，避免重排完全推翻召回
            base = 1.0 / (10 + item.rank)
            scored.append((coverage + position_bonus + title_bonus + base, item))

        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [
            RetrievedChunk(chunk=item.chunk, score=round(score, 6), rank=i + 1, retriever=f"{item.retriever}|rerank")
            for i, (score, item) in enumerate(scored[:top_k])
        ]


class HybridRetriever:
    """检索编排入口。按 mode 组合上面的组件，并产出 RetrievalTrace。"""

    def __init__(
        self,
        *,
        settings: Settings,
        vector: VectorRetriever | None,
        bm25: BM25Retriever | None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._vector = vector
        self._bm25 = bm25
        self._reranker = reranker or LexicalReranker()

    async def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        category: DocCategory | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        mode = mode or self._settings.retrieval_mode
        top_k = top_k or self._settings.top_k
        candidate_k = max(top_k, self._settings.candidate_k)
        trace = RetrievalTrace(query=query, mode=mode)

        vector_hits: list[RetrievedChunk] = []
        bm25_hits: list[RetrievedChunk] = []

        if mode in {"vector", "hybrid", "hybrid_rerank"} and self._vector:
            vector_hits = await self._vector.retrieve(query, top_k=candidate_k, category=category)
            trace.vector_candidates = [h.chunk_id for h in vector_hits]
        if mode in {"bm25", "hybrid", "hybrid_rerank"} and self._bm25:
            bm25_hits = await self._bm25.retrieve(query, top_k=candidate_k, category=category)
            trace.bm25_candidates = [h.chunk_id for h in bm25_hits]

        if mode == "vector":
            fused = vector_hits
        elif mode == "bm25":
            fused = bm25_hits
        else:
            fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=self._settings.rrf_k)
            trace.fused_order = [h.chunk_id for h in fused]

        if mode == "hybrid_rerank":
            fused = self._reranker.rerank(query, fused, top_k=candidate_k)
            trace.reranked_order = [h.chunk_id for h in fused]

        final = [h for h in fused if h.score >= self._settings.min_score][:top_k]
        trace.final_chunk_ids = [h.chunk_id for h in final]
        logger.info(
            "retrieval_done",
            extra={
                "mode": mode,
                "vector": len(vector_hits),
                "bm25": len(bm25_hits),
                "final": len(final),
            },
        )
        return final, trace


def build_context(chunks: list[RetrievedChunk], *, max_chars: int) -> tuple[str, list[str]]:
    """把检索结果拼成带 chunk_id 的上下文。返回 (上下文, 实际纳入的 chunk_id 列表)。

    整块纳入或整块丢弃，不截半块——否则 chunk_id 与内容会错配，引用校验就失去意义。
    """
    parts: list[str] = []
    used: list[str] = []
    total = 0
    for item in chunks:
        block = f"[{item.chunk_id}] 《{item.chunk.title}》\n{item.chunk.content.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(item.chunk_id)
        total += len(block)
    return "\n".join(parts), used


def chunks_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}
