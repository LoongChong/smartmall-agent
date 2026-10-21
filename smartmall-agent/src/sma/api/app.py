"""FastAPI 应用装配。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..agent.service import ChatService
from ..config import Settings, get_settings
from ..errors import SMAError
from ..logging_setup import get_logger, get_trace_id, new_trace_id, set_trace_id, setup_logging
from ..persistence.session_store import SessionStore
from .routes import router

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, service: ChatService | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        own = service is None
        if service is not None:
            app.state.service = service
        else:
            store = await SessionStore.open(settings.db_path, max_turns=settings.session_max_turns)
            app.state.store = store
            app.state.service = ChatService(settings=settings, session_store=store)
            # 启动时确保索引就绪：增量模式下已索引的文档会被跳过，成本很低
            try:
                report = await app.state.service.reindex(mode="incremental")
                logger.info("startup_index", extra=report.to_dict())
            except Exception:  # noqa: BLE001 - 索引失败不应阻止服务启动，但必须记录
                logger.exception("startup_index_failed")
        app.state.settings = settings
        try:
            yield
        finally:
            if own:
                await app.state.service.aclose()
                await app.state.store.close()

    app = FastAPI(
        title="SmartMall Agent",
        version=__version__,
        description="智能电商客服 Agent：混合检索 RAG + 业务工具 + 服务端权限与规则校验",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        trace_id = set_trace_id(request.headers.get("X-Trace-Id") or new_trace_id())
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(SMAError)
    async def sma_error_handler(_: Request, exc: SMAError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.error("api_error", extra={"code": exc.code, "error": exc.message})
        else:
            logger.warning("api_client_error", extra={"code": exc.code, "error": exc.message})
        return JSONResponse(status_code=exc.http_status, content={**exc.to_dict(), "trace_id": get_trace_id()})

    app.include_router(router)
    return app


app = create_app()
