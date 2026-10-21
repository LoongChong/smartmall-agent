"""API 路由。

身份模型（见 docs/SECURITY.md）：
用户身份来自 ``X-User-Id`` 请求头，由**网关/认证层**注入，不是客户端随便填的。
本项目不实现登录体系，但明确了这个边界：
- 聊天里自述的身份（"我是张三，帮我查订单 SOxxx"）**永远不会**被当作身份；
- 没有 ``X-User-Id`` 的请求只能使用无需鉴权的工具（商品查询、知识库问答）。

这是演示项目的简化，README 与 SECURITY 都如实标注了。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from .. import __version__
from ..agent.service import ChatService
from ..config import Settings
from ..errors import UnauthorizedError
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    KnowledgeStatusResponse,
    ReindexRequest,
    ReindexResponse,
)

router = APIRouter()


def get_service(request: Request) -> ChatService:
    return request.app.state.service  # type: ignore[no-any-return]


def get_config(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


ServiceDep = Annotated[ChatService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_config)]


def require_admin(settings: SettingsDep, x_admin_token: Annotated[str | None, Header()] = None) -> None:
    """管理接口鉴权。未配置 token 时直接拒绝，避免"默认开放"。"""
    if not settings.reindex_requires_admin:
        return
    if not settings.admin_token_value:
        raise UnauthorizedError("管理接口未启用：请配置 SMA_API_ADMIN_TOKEN")
    if x_admin_token != settings.admin_token_value:
        raise UnauthorizedError("管理员令牌无效")


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(service: ServiceDep, settings: SettingsDep) -> HealthResponse:
    count = service.vector_store.count()
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        knowledge_ready=count > 0,
        chunk_count=count,
    )


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    payload: ChatRequest,
    service: ServiceDep,
    x_user_id: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    answer = await service.chat(
        question=payload.question,
        session_id=payload.session_id,
        user_id=x_user_id,
        authenticated=bool(x_user_id),
    )
    return ChatResponse(
        session_id=answer.session_id,
        trace_id=answer.trace_id,
        status=answer.status,
        intent=answer.intent,
        answer=answer.answer,
        citations=answer.citations,
        dropped_citations=answer.dropped_citations,
        tool_calls=answer.tool_calls,
        retrieval_trace=answer.retrieval_trace if payload.include_trace else None,
        needs_human=answer.needs_human,
        elapsed_ms=answer.elapsed_ms,
    )


@router.get("/api/v1/knowledge/status", response_model=KnowledgeStatusResponse, tags=["knowledge"])
async def knowledge_status(service: ServiceDep) -> KnowledgeStatusResponse:
    return KnowledgeStatusResponse(**service.knowledge_status())


@router.post(
    "/api/v1/knowledge/reindex",
    response_model=ReindexResponse,
    tags=["knowledge"],
    dependencies=[Depends(require_admin)],
)
async def reindex(payload: ReindexRequest, service: ServiceDep) -> ReindexResponse:
    """重建知识库。

    **需要管理员令牌**：全量重建是高成本操作（嵌入全部文档），
    不能让任意用户触发（会被当成 DoS 放大器）。
    """
    report = await service.reindex(mode=payload.mode)
    return ReindexResponse(
        ok=report.ok,
        mode=report.mode,
        docs_total=report.docs_total,
        docs_indexed=report.docs_indexed,
        docs_skipped=report.docs_skipped,
        docs_removed=report.docs_removed,
        chunks_written=report.chunks_written,
        collection_count=report.collection_count,
        elapsed_ms=report.elapsed_ms,
        failures=report.failures,
    )


@router.get("/api/v1/tools", tags=["system"], dependencies=[Depends(require_admin)])
async def list_tools(service: ServiceDep) -> list[dict]:
    """列出已注册工具及其权限属性。用于审计 Agent 的能力边界。"""
    return service.registry.describe()
