"""回答校验：引用真实性 + 无依据数字检测。

两条校验：

1. **引用校验**：回答中的 chunk_id 必须真实出现在本次检索的上下文里。
   编造的引用被剔除并记入 ``dropped_citations``；
   如果剔除后一条引用都不剩，状态降级为 ``no_answer``——
   因为"声称有依据但一条真引用都没有"比"说不知道"更危险。

2. **数字幻觉检测**：回答里出现的数字（价格、天数、费用）如果在检索上下文中
   完全找不到，就是可疑的。这类情况不直接拒绝回答（数字可能来自工具返回或是合理改写），
   但会在日志里告警，并记录到回答的审计字段里。
"""

from __future__ import annotations

import re

from ..logging_setup import get_logger
from ..models import AnswerStatus, Chunk, Citation, RetrievedChunk

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def validate_citations(
    answer: str,
    raw_citations: list[str],
    retrieved: list[RetrievedChunk],
    *,
    status: AnswerStatus,
) -> tuple[AnswerStatus, list[Citation], list[str]]:
    """返回 (修正后的状态, 真实引用列表, 被剔除的引用 ID)。"""
    index: dict[str, Chunk] = {item.chunk_id: item.chunk for item in retrieved}

    kept: list[str] = []
    dropped: list[str] = []
    for cid in dict.fromkeys(raw_citations):
        (kept if cid in index else dropped).append(cid)

    citations = [
        Citation(
            chunk_id=cid,
            doc_id=index[cid].doc_id,
            title=index[cid].title,
            category=index[cid].category,
            snippet=index[cid].content.strip()[:150],
        )
        for cid in kept
    ]

    if dropped:
        logger.warning("citation_dropped", extra={"count": len(dropped), "ids": dropped[:5]})

    if status == AnswerStatus.ANSWERED and not citations:
        logger.warning("answer_without_citation", extra={"answer": answer[:100]})
        return AnswerStatus.NO_ANSWER, [], dropped

    return status, citations, dropped


def find_unsupported_numbers(answer: str, retrieved: list[RetrievedChunk]) -> list[str]:
    """找出回答中出现、但检索上下文里没有的数字。

    只做告警不做拦截：模型可能做了合理的单位换算或改写。
    但如果这个列表频繁非空，说明提示词约束不够。
    """
    context = " ".join(item.chunk.content for item in retrieved)
    context_numbers = set(_NUMBER_RE.findall(context))
    answer_numbers = _NUMBER_RE.findall(answer)
    unsupported = [n for n in dict.fromkeys(answer_numbers) if n not in context_numbers]
    if unsupported:
        logger.warning("unsupported_numbers", extra={"numbers": unsupported[:10]})
    return unsupported


def citation_support_rate(answers: list[tuple[AnswerStatus, int]]) -> float:
    """批量评测用：应当有引用的回答中，实际有引用的比例。"""
    answerable = [(s, n) for s, n in answers if s == AnswerStatus.ANSWERED]
    if not answerable:
        return 0.0
    return round(sum(1 for _, n in answerable if n > 0) / len(answerable), 4)
