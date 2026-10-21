"""评测指标。

口径都写清楚，避免"同一个指标不同算法"造成的数字不可比。
"""

from __future__ import annotations

from ..models import RetrievedChunk


def recall_at_k(retrieved_docs: list[str], expected_docs: set[str], k: int) -> float:
    """Recall@K = 前 K 个结果覆盖的期望文档数 / 期望文档总数。

    按**文档**粒度而不是块粒度：切块边界会随 chunk_size 变化，
    块粒度的标注无法在参数调整后复用。
    """
    if not expected_docs:
        return 0.0
    top = set(retrieved_docs[:k])
    return round(len(top & expected_docs) / len(expected_docs), 4)


def mrr(retrieved_docs: list[str], expected_docs: set[str]) -> float:
    """MRR = 1 / 第一个命中期望文档的排名。没有命中则为 0。"""
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in expected_docs:
            return round(1.0 / rank, 4)
    return 0.0


def precision_at_k(retrieved_docs: list[str], expected_docs: set[str], k: int) -> float:
    """Precision@K = 前 K 个结果中命中期望文档的比例（按去重后的文档算）。"""
    top = list(dict.fromkeys(retrieved_docs[:k]))
    if not top:
        return 0.0
    return round(sum(1 for d in top if d in expected_docs) / len(top), 4)


def dedup_docs(hits: list[RetrievedChunk]) -> list[str]:
    """把块级检索结果折叠成文档级列表，保持顺序。"""
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))


def percentile(values: list[float], p: float) -> float:
    """最近邻插值分位数。样本量小时比线性插值更不容易产生误导性的"精确值"。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[idx], 2)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
