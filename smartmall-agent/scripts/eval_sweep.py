"""多 K 值评测扫描。

动机：知识库只有 6 篇文档，Recall@4 对所有检索模式都是 1.0（天花板效应），
看不出差异。K=1/2 才能暴露排序质量的区别——这正是混合检索与重排要解决的问题。

用法：uv run python scripts/eval_sweep.py > docs/benchmarks/sweep_<date>.json
"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sma.agent.service import ChatService  # noqa: E402
from sma.config import get_settings  # noqa: E402
from sma.eval.runner import MODES, evaluate_retrieval  # noqa: E402
from sma.logging_setup import setup_logging  # noqa: E402
from sma.persistence.session_store import SessionStore  # noqa: E402


async def main() -> int:
    settings = get_settings()
    setup_logging("ERROR", json_output=False)
    store = await SessionStore.open(settings.db_path)
    service = ChatService(settings=settings, session_store=store)
    try:
        await service.reindex(mode="incremental")
        table: dict[str, dict[str, dict[str, float]]] = {}
        for k in (1, 2, 3, 4):
            results = await evaluate_retrieval(service, modes=MODES, k=k)
            table[f"k={k}"] = {
                mode: {
                    "recall": data[f"recall@{k}"],
                    "mrr": data["mrr"],
                    "precision": data[f"precision@{k}"],
                    "p50_ms": data["latency_p50_ms"],
                }
                for mode, data in results.items()
            }
        print(
            json.dumps(
                {
                    "environment": {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "embedding_provider": settings.embedding_provider,
                        "chunk_size": settings.chunk_size,
                        "rrf_k": settings.rrf_k,
                    },
                    "note": (
                        "知识库仅 6 篇文档，K=4 时所有模式 Recall 均为 1.0（天花板效应），"
                        "区分度体现在 K=1/2 的 Recall 与各 K 的 Precision 上。"
                        "embedding_provider=hash 是词面哈希向量，非语义向量。"
                    ),
                    "results": table,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        await service.aclose()
        await store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
