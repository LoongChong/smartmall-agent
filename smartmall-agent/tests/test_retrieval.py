"""检索：向量、BM25、混合融合、重排、上下文构造。"""

from __future__ import annotations

import pytest

from sma.agent.service import ChatService
from sma.config import Settings
from sma.errors import KnowledgeBaseEmptyError
from sma.models import Chunk, DocCategory, RetrievedChunk
from sma.rag.bm25 import BM25Index, tokenize
from sma.rag.embedding import HashEmbedding, cosine
from sma.rag.retriever import (
    BM25Retriever,
    LexicalReranker,
    VectorRetriever,
    build_context,
    reciprocal_rank_fusion,
)
from sma.rag.vector_store import ChromaVectorStore


def make_chunk(cid: str, content: str, title: str = "T") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=cid.split("#")[0],
        content=content,
        seq=0,
        title=title,
        category=DocCategory.GENERAL_FAQ,
    )


def make_hit(cid: str, rank: int, retriever: str = "x", content: str = "内容") -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(cid, content), score=1.0 / rank, rank=rank, retriever=retriever)


# ---------------- 分词与 Embedding ----------------


def test_tokenize_filters_stopwords() -> None:
    tokens = tokenize("请问一下这个商品的保修期是多久呢")
    assert "保修期" in tokens or "保修" in tokens
    assert "的" not in tokens and "呢" not in tokens


async def test_hash_embedding_is_deterministic() -> None:
    e = HashEmbedding(dim=128)
    a = await e.embed_one("退货政策")
    b = await e.embed_one("退货政策")
    assert a == b
    assert len(a) == 128
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6, "应为 L2 归一化向量"


async def test_hash_embedding_similarity_reflects_overlap() -> None:
    e = HashEmbedding(dim=256)
    base = await e.embed_one("7天无理由退货政策说明")
    near = await e.embed_one("无理由退货政策")
    far = await e.embed_one("蓝牙耳机降噪深度")
    assert cosine(base, near) > cosine(base, far)


async def test_embedding_empty_text() -> None:
    e = HashEmbedding(dim=64)
    vec = await e.embed_one("")
    assert vec == [0.0] * 64


# ---------------- BM25 ----------------


def test_bm25_ranks_relevant_chunk_first() -> None:
    chunks = [
        make_chunk("a#0", "退换货政策：7 天无理由退货，自签收之日起算"),
        make_chunk("b#0", "蓝牙耳机降噪深度最高 48dB，续航 8 小时"),
        make_chunk("c#0", "发票可以开电子普通发票或增值税专用发票"),
    ]
    index = BM25Index(chunks)
    hits = index.search("无理由退货几天", top_k=3)
    assert hits[0][0].chunk_id == "a#0"


def test_bm25_empty_query_returns_nothing() -> None:
    index = BM25Index([make_chunk("a#0", "内容")])
    assert index.search("的了吗", top_k=3) == []


def test_bm25_on_empty_corpus() -> None:
    index = BM25Index([])
    assert index.size == 0
    assert index.search("任意", top_k=3) == []


# ---------------- RRF 融合 ----------------


def test_rrf_rewards_items_found_by_both() -> None:
    vector = [make_hit("a#0", 1), make_hit("b#0", 2), make_hit("c#0", 3)]
    bm25 = [make_hit("c#0", 1), make_hit("b#0", 2), make_hit("d#0", 3)]

    fused = reciprocal_rank_fusion([vector, bm25], k=60)
    ids = [f.chunk_id for f in fused]

    # b 和 c 被两路都召回，应排在只被一路召回的 a、d 之前
    assert ids.index("b#0") < ids.index("a#0")
    assert ids.index("c#0") < ids.index("d#0")
    assert len(ids) == len(set(ids)), "融合结果必须去重"


def test_rrf_marks_multi_retriever_source() -> None:
    fused = reciprocal_rank_fusion([[make_hit("a#0", 1, "vector")], [make_hit("a#0", 1, "bm25")]], k=60)
    assert fused[0].retriever == "bm25+vector"


def test_rrf_handles_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    single = reciprocal_rank_fusion([[make_hit("a#0", 1)], []])
    assert [s.chunk_id for s in single] == ["a#0"]


def test_rrf_is_scale_invariant() -> None:
    """RRF 只看排名：一路分数放大一万倍也不影响融合结果。"""
    normal = [make_hit("a#0", 1), make_hit("b#0", 2)]
    inflated = [
        RetrievedChunk(chunk=make_chunk("a#0", "x"), score=99999.0, rank=1, retriever="bm25"),
        RetrievedChunk(chunk=make_chunk("b#0", "y"), score=88888.0, rank=2, retriever="bm25"),
    ]
    assert [f.chunk_id for f in reciprocal_rank_fusion([normal, inflated])] == ["a#0", "b#0"]


# ---------------- Reranker ----------------


def test_reranker_prefers_higher_term_coverage() -> None:
    candidates = [
        make_hit("low#0", 1, content="这里只提到退货两个字，其余无关内容很多很多"),
        make_hit("high#0", 2, content="7 天无理由退货的起算点是签收时间，退货运费由买家承担"),
    ]
    reranked = LexicalReranker().rerank("无理由退货 起算点 签收", candidates, top_k=2)
    assert reranked[0].chunk_id == "high#0", "覆盖更多查询词的块应被提到前面"
    assert reranked[0].retriever.endswith("|rerank")


def test_reranker_with_empty_query_keeps_order() -> None:
    candidates = [make_hit("a#0", 1), make_hit("b#0", 2)]
    assert [c.chunk_id for c in LexicalReranker().rerank("的了吗", candidates, top_k=2)] == ["a#0", "b#0"]


# ---------------- 上下文构造 ----------------


def test_build_context_includes_chunk_ids() -> None:
    hits = [make_hit("doc_a#0", 1, content="内容一"), make_hit("doc_b#0", 2, content="内容二")]
    context, used = build_context(hits, max_chars=1000)
    assert "[doc_a#0]" in context and "[doc_b#0]" in context
    assert used == ["doc_a#0", "doc_b#0"]


def test_build_context_drops_whole_chunks_not_halves() -> None:
    hits = [make_hit(f"d#{i}", i + 1, content="长内容" * 100) for i in range(5)]
    context, used = build_context(hits, max_chars=400)
    assert len(used) < 5
    # 每个被纳入的 id 都必须有完整的块，不能出现截半
    for cid in used:
        assert f"[{cid}]" in context


# ---------------- 端到端检索 ----------------


async def test_vector_retriever_on_empty_store(settings: Settings, embedder: HashEmbedding) -> None:
    store = ChromaVectorStore(persist_dir=settings.chroma_dir, collection_name="empty_kb")
    with pytest.raises(KnowledgeBaseEmptyError):
        await VectorRetriever(store, embedder).retrieve("任意问题", top_k=3)


async def test_bm25_retriever_on_empty_index() -> None:
    with pytest.raises(KnowledgeBaseEmptyError):
        await BM25Retriever(BM25Index([])).retrieve("任意问题", top_k=3)


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "hybrid_rerank"])
async def test_all_modes_find_return_policy(service: ChatService, mode: str) -> None:
    hits, trace = await service.retrieve_only("7 天无理由退货从什么时候开始算", mode=mode, top_k=4)
    assert hits, f"模式 {mode} 应有召回"
    assert trace.mode == mode
    docs = {h.chunk.doc_id for h in hits}
    assert "aftersale_return_policy" in docs


async def test_hybrid_trace_records_both_paths(service: ChatService) -> None:
    _, trace = await service.retrieve_only("退货运费谁承担", mode="hybrid", top_k=4)
    assert trace.vector_candidates, "应记录向量候选"
    assert trace.bm25_candidates, "应记录 BM25 候选"
    assert trace.fused_order, "应记录融合顺序"


async def test_rerank_mode_records_rerank_order(service: ChatService) -> None:
    _, trace = await service.retrieve_only("保修期多久", mode="hybrid_rerank", top_k=4)
    assert trace.reranked_order
    assert trace.final_chunk_ids


async def test_colloquial_query_hits_right_doc(service: ChatService) -> None:
    """口语化表达也要能召回：BM25 分词在这里的贡献大于词面向量。"""
    hits, _ = await service.retrieve_only("啥时候能发货啊", mode="hybrid", top_k=4)
    assert "general_faq" in {h.chunk.doc_id for h in hits}


async def test_retrieval_respects_top_k(service: ChatService) -> None:
    hits, _ = await service.retrieve_only("退货", mode="hybrid", top_k=2)
    assert len(hits) <= 2
