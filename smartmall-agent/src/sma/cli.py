"""命令行入口。

uv run sma index [--full]          # 构建知识库
uv run sma ask "问题" [--user U001] # 问一个问题
uv run sma retrieve "查询" --mode hybrid  # 只看检索结果
uv run sma eval                    # 跑评测集
uv run sma serve                   # 启动服务
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .agent.service import ChatService
from .config import get_settings
from .logging_setup import setup_logging
from .persistence.session_store import SessionStore


async def _with_service(fn, *, log_json: bool = False):  # type: ignore[no-untyped-def]
    settings = get_settings()
    setup_logging(settings.log_level, log_json)
    store = await SessionStore.open(settings.db_path, max_turns=settings.session_max_turns)
    service = ChatService(settings=settings, session_store=store)
    try:
        return await fn(service, settings)
    finally:
        await service.aclose()
        await store.close()


async def _index(full: bool) -> int:
    async def run(service: ChatService, _settings) -> int:  # type: ignore[no-untyped-def]
        report = await service.reindex(mode="full" if full else "incremental")
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1

    return await _with_service(run)


async def _ask(question: str, user_id: str | None) -> int:
    async def run(service: ChatService, _settings) -> int:  # type: ignore[no-untyped-def]
        await service.reindex(mode="incremental")
        answer = await service.chat(question=question, user_id=user_id, authenticated=bool(user_id))
        print(f"\n[意图] {answer.intent.value}   [状态] {answer.status.value}   [{answer.elapsed_ms}ms]")
        print(f"\n{answer.answer}\n")
        if answer.citations:
            print("来源：")
            for c in answer.citations:
                print(f"  - [{c.chunk_id}] 《{c.title}》")
        if answer.dropped_citations:
            print(f"被剔除的伪造引用: {answer.dropped_citations}")
        if answer.tool_calls:
            print("工具调用：")
            for t in answer.tool_calls:
                print(f"  - {t.tool} -> {t.status.value} ({t.elapsed_ms}ms) {t.result_summary[:80]}")
        return 0

    return await _with_service(run)


async def _retrieve(query: str, mode: str, top_k: int) -> int:
    async def run(service: ChatService, _settings) -> int:  # type: ignore[no-untyped-def]
        await service.reindex(mode="incremental")
        hits, trace = await service.retrieve_only(query, mode=mode, top_k=top_k)
        n_vec, n_bm25 = len(trace.vector_candidates), len(trace.bm25_candidates)
        print(f"\n模式: {trace.mode}   向量候选: {n_vec}   BM25 候选: {n_bm25}\n")
        for h in hits:
            print(f"#{h.rank} [{h.chunk_id}] score={h.score:.4f} via={h.retriever}")
            print(f"    {h.chunk.content[:100].replace(chr(10), ' ')}...")
        return 0

    return await _with_service(run)


async def _eval() -> int:
    from .eval.runner import run_evaluation

    async def run(service: ChatService, _settings) -> int:  # type: ignore[no-untyped-def]
        await service.reindex(mode="incremental")
        report = await run_evaluation(service)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    return await _with_service(run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sma", description="SmartMall Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="构建知识库")
    p_index.add_argument("--full", action="store_true", help="全量重建")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--user", dest="user_id", default=None, help="模拟已认证用户，如 U001")

    p_ret = sub.add_parser("retrieve", help="只看检索结果")
    p_ret.add_argument("query")
    p_ret.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    p_ret.add_argument("--top-k", type=int, default=5)

    sub.add_parser("eval", help="运行评测集")

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "index":
        return asyncio.run(_index(args.full))
    if args.command == "ask":
        return asyncio.run(_ask(args.question, args.user_id))
    if args.command == "retrieve":
        return asyncio.run(_retrieve(args.query, args.mode, args.top_k))
    if args.command == "eval":
        return asyncio.run(_eval())
    if args.command == "serve":
        import uvicorn

        uvicorn.run("sma.api.app:app", host=args.host, port=args.port, log_config=None)
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
