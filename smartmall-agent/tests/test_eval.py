"""评测集回归。

这些测试保证：评测脚本本身可用、固定用例集不退化。
指标的具体数值写在 docs/EVALUATION.md，这里只断言**不退化的下界**，
避免每次微调参数都要改测试。
"""

from __future__ import annotations

from sma.agent.service import ChatService
from sma.eval.dataset import ALL_CASES, BEHAVIOR_CASES, RETRIEVAL_CASES, CaseType
from sma.eval.metrics import mrr, precision_at_k, recall_at_k
from sma.eval.runner import evaluate_behavior, evaluate_retrieval, run_evaluation

# ---------------- 指标函数 ----------------


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert recall_at_k(["x"], {"a"}, 1) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_recall_respects_k() -> None:
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_mrr() -> None:
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    assert precision_at_k([], {"a"}, 2) == 0.0


# ---------------- 数据集完整性 ----------------


def test_dataset_covers_required_types() -> None:
    """需求点名要覆盖的用例类型必须都在。"""
    types = {c.case_type for c in ALL_CASES}
    required = {
        CaseType.PRODUCT_ATTR,
        CaseType.MULTI_CONDITION,
        CaseType.AFTERSALE_POLICY,
        CaseType.PAYMENT,
        CaseType.COLLOQUIAL,
        CaseType.NO_ANSWER,
        CaseType.AMBIGUOUS,
        CaseType.TOOL_CALL,
        CaseType.UNAUTHORIZED,
        CaseType.PROMPT_INJECTION,
    }
    assert required <= types


def test_case_ids_unique() -> None:
    ids = [c.case_id for c in ALL_CASES]
    assert len(ids) == len(set(ids))


def test_retrieval_cases_have_expected_docs() -> None:
    assert all(c.expected_docs for c in RETRIEVAL_CASES), "检索用例必须有标注文档"


def test_behavior_cases_have_expected_status() -> None:
    valid = {"answered", "no_answer", "clarify", "forbidden", "escalated"}
    assert all(c.expected_status in valid for c in BEHAVIOR_CASES)


# ---------------- 端到端评测 ----------------


async def test_retrieval_evaluation_runs(service: ChatService) -> None:
    results = await evaluate_retrieval(service, modes=("bm25", "hybrid"), k=4)
    for mode in ("bm25", "hybrid"):
        assert results[mode]["cases"] == len(RETRIEVAL_CASES)
        assert len(results[mode]["details"]) == len(RETRIEVAL_CASES)
        assert 0.0 <= results[mode]["recall@4"] <= 1.0


async def test_retrieval_quality_does_not_regress(service: ChatService) -> None:
    """下界断言：标注集上 hybrid 的 Recall@4 不应低于 0.8、MRR 不低于 0.7。

    具体数值见 docs/EVALUATION.md（当前实测 1.0 / 1.0）。
    这里留出余量，避免正常的参数调整导致测试频繁失败。
    """
    results = await evaluate_retrieval(service, modes=("hybrid",), k=4)
    assert results["hybrid"]["recall@4"] >= 0.8
    assert results["hybrid"]["mrr"] >= 0.7


async def test_behavior_evaluation_all_pass(service: ChatService) -> None:
    """行为用例必须全通过：它们断言的是安全与正确性，没有"大致正确"一说。"""
    report = await evaluate_behavior(service)
    failures = [d for d in report["details"] if not d["pass"]]
    assert not failures, f"行为用例失败: {[(f['case_id'], f['actual_status']) for f in failures]}"
    assert report["pass_rate"] == 1.0


async def test_full_evaluation_report_shape(service: ChatService) -> None:
    report = await run_evaluation(service, k=4)
    assert set(report["retrieval_summary"]) == {"vector", "bm25", "hybrid", "hybrid_rerank"}
    assert report["dataset"]["total"] == len(ALL_CASES)
    assert report["behavior"]["pass_rate"] == 1.0
    assert "hash" in report["note"], "报告必须声明 hash embedding 的局限"
    assert report["environment"]["embedding_provider"] == "hash"


async def test_evaluation_is_reproducible(service: ChatService) -> None:
    """同样的配置跑两次，检索指标必须完全一致。"""
    first = await evaluate_retrieval(service, modes=("hybrid",), k=4)
    second = await evaluate_retrieval(service, modes=("hybrid",), k=4)
    assert first["hybrid"]["recall@4"] == second["hybrid"]["recall@4"]
    assert first["hybrid"]["mrr"] == second["hybrid"]["mrr"]
    assert [d["retrieved"] for d in first["hybrid"]["details"]] == [d["retrieved"] for d in second["hybrid"]["details"]]
