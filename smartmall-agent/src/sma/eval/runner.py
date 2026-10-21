"""评测执行器。

两部分：
1. **检索对比**：同一组标注用例分别用 vector / bm25 / hybrid / hybrid_rerank 跑一遍，
   输出 Recall@K、MRR、Precision@K。这是可复现的硬指标。
2. **行为检查**：跑行为类用例，检查回答状态是否符合预期
   （拒答、澄清、越权拦截、高风险转人工）。

所有数字由这个脚本现场跑出来，不预填任何值。
"""

from __future__ import annotations

import platform
import time
from typing import Any

from ..logging_setup import get_logger
from .dataset import ALL_CASES, BEHAVIOR_CASES, RETRIEVAL_CASES, EvalCase
from .metrics import dedup_docs, mean, mrr, percentile, precision_at_k, recall_at_k

logger = get_logger(__name__)

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


async def evaluate_retrieval(service: Any, *, modes: tuple[str, ...] = MODES, k: int = 4) -> dict[str, Any]:
    """对比不同检索模式。返回每个模式的聚合指标与逐条明细。"""
    results: dict[str, Any] = {}
    for mode in modes:
        recalls: list[float] = []
        mrrs: list[float] = []
        precisions: list[float] = []
        latencies: list[float] = []
        details: list[dict[str, Any]] = []

        for case in RETRIEVAL_CASES:
            expected = set(case.expected_docs)
            started = time.perf_counter()
            hits, _ = await service.retrieve_only(case.question, mode=mode, top_k=k)
            elapsed_ms = (time.perf_counter() - started) * 1000

            docs = dedup_docs(hits)
            r = recall_at_k(docs, expected, k)
            m = mrr(docs, expected)
            p = precision_at_k(docs, expected, k)
            recalls.append(r)
            mrrs.append(m)
            precisions.append(p)
            latencies.append(elapsed_ms)
            details.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "type": case.case_type.value,
                    "expected": sorted(expected),
                    "retrieved": docs,
                    "recall@k": r,
                    "mrr": m,
                }
            )

        results[mode] = {
            f"recall@{k}": mean(recalls),
            "mrr": mean(mrrs),
            f"precision@{k}": mean(precisions),
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "cases": len(RETRIEVAL_CASES),
            "details": details,
        }
        logger.info(
            "eval_mode_done",
            extra={"mode": mode, "recall": results[mode][f"recall@{k}"], "mrr": results[mode]["mrr"]},
        )
    return results


async def evaluate_behavior(service: Any) -> dict[str, Any]:
    """行为用例：检查回答状态是否符合预期。"""
    passed = 0
    details: list[dict[str, Any]] = []
    latencies: list[float] = []

    for case in BEHAVIOR_CASES:
        answer = await service.chat(
            question=case.question,
            session_id=f"eval_{case.case_id}",
            user_id=case.user_id,
            authenticated=bool(case.user_id),
        )
        latencies.append(float(answer.elapsed_ms))
        ok = answer.status.value == case.expected_status
        passed += int(ok)
        details.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "type": case.case_type.value,
                "expected_status": case.expected_status,
                "actual_status": answer.status.value,
                "intent": answer.intent.value,
                "pass": ok,
                "citations": len(answer.citations),
                "dropped_citations": len(answer.dropped_citations),
                "needs_human": answer.needs_human,
                "note": case.note,
            }
        )

    return {
        "cases": len(BEHAVIOR_CASES),
        "passed": passed,
        "pass_rate": round(passed / len(BEHAVIOR_CASES), 4) if BEHAVIOR_CASES else 0.0,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "details": details,
    }


async def run_evaluation(service: Any, *, k: int = 4) -> dict[str, Any]:
    started = time.perf_counter()
    retrieval = await evaluate_retrieval(service, k=k)
    behavior = await evaluate_behavior(service)

    summary = {
        mode: {
            f"recall@{k}": data[f"recall@{k}"],
            "mrr": data["mrr"],
            f"precision@{k}": data[f"precision@{k}"],
            "latency_p50_ms": data["latency_p50_ms"],
        }
        for mode, data in retrieval.items()
    }

    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "llm_provider": service.settings.llm_provider,
            "embedding_provider": service.settings.embedding_provider,
            "chunk_size": service.settings.chunk_size,
            "top_k": k,
        },
        "dataset": {
            "retrieval_cases": len(RETRIEVAL_CASES),
            "behavior_cases": len(BEHAVIOR_CASES),
            "total": len(ALL_CASES),
        },
        "retrieval_summary": summary,
        "retrieval_detail": retrieval,
        "behavior": behavior,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "note": (
            "embedding_provider=hash 时向量检索是词面哈希而非语义向量，"
            "其指标不代表真实 embedding 模型的表现；对比结论仅在同一配置下有效。"
        ),
    }


def failed_cases(report: dict[str, Any]) -> list[EvalCase]:
    failed_ids = {d["case_id"] for d in report["behavior"]["details"] if not d["pass"]}
    return [c for c in BEHAVIOR_CASES if c.case_id in failed_ids]
