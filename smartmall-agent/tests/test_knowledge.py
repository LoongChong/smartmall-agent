"""文档加载、切块、索引构建与重复执行。"""

from __future__ import annotations

from pathlib import Path

import pytest

from sma.config import Settings
from sma.errors import IndexingError
from sma.knowledge.chunker import chunk_document, split_long_text, split_sections
from sma.knowledge.indexer import KnowledgeIndexer
from sma.knowledge.loader import load_directory, load_document, parse_frontmatter
from sma.models import DocCategory, DocumentMeta
from sma.rag.vector_store import ChromaVectorStore

# ---------------- 加载 ----------------


def test_parse_frontmatter() -> None:
    meta, body = parse_frontmatter("---\ndoc_id: a\ntitle: 标题\ncategory: payment_faq\n---\n正文内容")
    assert meta == {"doc_id": "a", "title": "标题", "category": "payment_faq"}
    assert body.strip() == "正文内容"


def test_parse_frontmatter_absent() -> None:
    meta, body = parse_frontmatter("# 没有 frontmatter\n正文")
    assert meta == {}
    assert body.startswith("# 没有")


def test_load_document_extracts_metadata(knowledge_dir: Path) -> None:
    meta, body = load_document(knowledge_dir / "product_spec_p1001.md")
    assert meta.doc_id == "product_spec_p1001"
    assert meta.category == DocCategory.PRODUCT_SPEC
    assert meta.product_id == "P1001"
    assert meta.version and len(meta.version) == 16
    assert "星云 Pro 14" in body


def test_version_changes_with_content() -> None:
    assert DocumentMeta.version_of("a") != DocumentMeta.version_of("b")
    assert DocumentMeta.version_of("a") == DocumentMeta.version_of("a")


def test_load_directory_all_docs(knowledge_dir: Path) -> None:
    docs = load_directory(knowledge_dir)
    assert len(docs) == 6
    categories = {m.category for m, _ in docs}
    # 需求要求至少覆盖商品参数、售后政策、支付 FAQ、常见问题
    assert categories == {
        DocCategory.PRODUCT_SPEC,
        DocCategory.AFTERSALE_POLICY,
        DocCategory.PAYMENT_FAQ,
        DocCategory.GENERAL_FAQ,
    }


def test_load_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(IndexingError, match="不存在"):
        load_directory(tmp_path / "nope")


def test_load_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(IndexingError, match="没有 .md"):
        load_directory(tmp_path / "empty")


def test_load_rejects_duplicate_doc_id(tmp_path: Path) -> None:
    d = tmp_path / "dup"
    d.mkdir()
    for name in ("a.md", "b.md"):
        (d / name).write_text("---\ndoc_id: same\ntitle: T\ncategory: general_faq\n---\n内容", encoding="utf-8")
    with pytest.raises(IndexingError, match="重复的 doc_id"):
        load_directory(d)


def test_load_rejects_empty_body(tmp_path: Path) -> None:
    p = tmp_path / "e.md"
    p.write_text("---\ndoc_id: e\ntitle: T\ncategory: general_faq\n---\n   \n", encoding="utf-8")
    with pytest.raises(IndexingError, match="正文为空"):
        load_document(p)


# ---------------- 切块 ----------------


def test_split_sections_by_heading() -> None:
    sections = split_sections("# 标题\n引言\n## A\n内容A\n## B\n内容B")
    titles = [t for t, _ in sections]
    assert "A" in titles and "B" in titles
    body_a = next(b for t, b in sections if t == "A")
    assert body_a == "内容A"


def test_split_long_text_with_overlap() -> None:
    text = "第一句。" * 200
    parts = split_long_text(text, chunk_size=100, overlap=20)
    assert len(parts) > 1
    assert all(len(p) <= 220 for p in parts)


def test_split_short_text_not_split() -> None:
    assert split_long_text("短文本", chunk_size=100, overlap=10) == ["短文本"]


def test_chunk_ids_are_stable(knowledge_dir: Path) -> None:
    meta, body = load_document(knowledge_dir / "aftersale_return_policy.md")
    first = chunk_document(meta, body, chunk_size=320, overlap=64)
    second = chunk_document(meta, body, chunk_size=320, overlap=64)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "aftersale_return_policy#0"


def test_chunk_carries_metadata(knowledge_dir: Path) -> None:
    meta, body = load_document(knowledge_dir / "product_spec_p1001.md")
    chunks = chunk_document(meta, body, chunk_size=320, overlap=64)
    assert all(c.doc_id == "product_spec_p1001" for c in chunks)
    assert all(c.product_id == "P1001" for c in chunks)
    assert all(c.category == DocCategory.PRODUCT_SPEC for c in chunks)
    # 标题被拼进正文，利于 BM25 与向量命中
    assert all(meta.title in c.content for c in chunks)


def test_chunk_section_boundary_preserved(knowledge_dir: Path) -> None:
    """ "不支持无理由退货的情况"必须独立成块，不与"7 天无理由"混在一起。"""
    meta, body = load_document(knowledge_dir / "aftersale_return_policy.md")
    chunks = chunk_document(meta, body, chunk_size=320, overlap=64)
    negative = [c for c in chunks if "不支持无理由退货的情况" in c.content]
    assert negative, "否定条款应该独立成块"
    assert "7 天的起算点" not in negative[0].content


# ---------------- 索引 ----------------


async def test_build_index(indexer: KnowledgeIndexer, vector_store: ChromaVectorStore) -> None:
    report = await indexer.build(mode="incremental")
    assert report.ok
    assert report.docs_total == 6
    assert report.docs_indexed == 6
    assert report.chunks_written > 0
    assert vector_store.count() == report.chunks_written


async def test_rebuild_is_idempotent(indexer: KnowledgeIndexer, vector_store: ChromaVectorStore) -> None:
    first = await indexer.build(mode="incremental")
    count_after_first = vector_store.count()

    second = await indexer.build(mode="incremental")

    assert second.docs_indexed == 0, "内容未变时不应重复索引"
    assert second.docs_skipped == first.docs_total
    assert second.chunks_written == 0
    assert vector_store.count() == count_after_first, "重复构建不得增加块数"


async def test_content_change_triggers_reindex(
    indexer: KnowledgeIndexer, vector_store: ChromaVectorStore, knowledge_dir: Path
) -> None:
    await indexer.build(mode="incremental")
    before = vector_store.doc_versions()["general_faq"]

    target = knowledge_dir / "general_faq.md"
    target.write_text(target.read_text(encoding="utf-8") + "\n## 新增小节\n这是新增的内容。\n", encoding="utf-8")
    report = await indexer.build(mode="incremental")

    assert report.docs_indexed == 1
    assert report.docs_skipped == 5
    assert vector_store.doc_versions()["general_faq"] != before


async def test_shrinking_document_removes_stale_chunks(
    indexer: KnowledgeIndexer, vector_store: ChromaVectorStore, knowledge_dir: Path
) -> None:
    """文档变短时旧块必须被删除，否则会残留孤儿块污染检索。"""
    await indexer.build(mode="incremental")
    before = len([c for c in vector_store.all_chunks() if c.doc_id == "general_faq"])
    assert before > 1

    target = knowledge_dir / "general_faq.md"
    target.write_text(
        "---\ndoc_id: general_faq\ntitle: 购物常见问题\ncategory: general_faq\n---\n# 精简\n只剩一句话。\n",
        encoding="utf-8",
    )
    await indexer.build(mode="incremental")

    after = len([c for c in vector_store.all_chunks() if c.doc_id == "general_faq"])
    assert after < before, "切块变少时旧块应被清理"


async def test_deleted_document_removed_from_index(
    indexer: KnowledgeIndexer, vector_store: ChromaVectorStore, knowledge_dir: Path
) -> None:
    await indexer.build(mode="incremental")
    (knowledge_dir / "general_faq.md").unlink()

    report = await indexer.build(mode="incremental")

    assert report.docs_removed == 1
    assert "general_faq" not in vector_store.doc_versions()


async def test_full_rebuild_resets_collection(indexer: KnowledgeIndexer, vector_store: ChromaVectorStore) -> None:
    await indexer.build(mode="incremental")
    report = await indexer.build(mode="full")
    assert report.docs_indexed == 6
    assert report.docs_skipped == 0
    assert vector_store.count() == report.chunks_written


async def test_index_failure_is_reported_not_swallowed(settings: Settings, vector_store: ChromaVectorStore) -> None:
    """单篇文档失败不中断整体，但必须出现在 failures 里。"""

    class BrokenEmbedder:
        dim = 8

        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding 服务挂了")

        async def embed_one(self, text: str) -> list[float]:
            raise RuntimeError("embedding 服务挂了")

        async def aclose(self) -> None:
            return None

    indexer = KnowledgeIndexer(settings=settings, store=vector_store, embedder=BrokenEmbedder())  # type: ignore[arg-type]
    report = await indexer.build(mode="incremental")

    assert not report.ok
    assert len(report.failures) == 6
    assert report.chunks_written == 0


async def test_build_bm25_from_store(indexer: KnowledgeIndexer) -> None:
    await indexer.build(mode="incremental")
    bm25 = indexer.build_bm25()
    assert bm25.size > 0
    assert len(bm25.idf) > 50
