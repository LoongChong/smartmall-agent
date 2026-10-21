"""ChatService：应用层编排。

职责：装配 RAG 组件与工具，驱动 LangGraph 图，维护会话。
每个图节点对应这里的一个方法，方法本身可以脱离 LangGraph 单测。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings
from ..errors import KnowledgeBaseEmptyError, LLMError, SMAError
from ..knowledge.indexer import IndexReport, KnowledgeIndexer
from ..llm.base import LLMClient, LLMMessage
from ..llm.factory import build_embedder, build_llm_client
from ..logging_setup import get_logger, get_trace_id, new_trace_id, set_session_id, set_trace_id
from ..models import (
    AnswerStatus,
    ChatAnswer,
    ChatTurn,
    Intent,
    IntentResult,
    RetrievalTrace,
    RetrievedChunk,
    ToolCallRecord,
    ToolStatus,
)
from ..persistence.session_store import SessionStore
from ..rag.bm25 import BM25Index
from ..rag.embedding import Embedder
from ..rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    LexicalReranker,
    VectorRetriever,
    build_context,
)
from ..rag.vector_store import ChromaVectorStore
from ..tools.base import ToolContext, ToolRegistry
from ..tools.business_tools import build_registry
from .guards import detect_identity_spoof, detect_injection, relevance_guard
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    CLARIFY_TEMPLATE,
    HIGH_RISK_ANSWER,
    INTENT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_ANSWER,
    TOOL_ANSWER_SYSTEM_PROMPT,
)
from .validator import find_unsupported_numbers, validate_citations
from .workflow import AgentState, build_agent_graph

logger = get_logger(__name__)

#: 意图 → 工具名。集中映射，避免模型直接指定工具名（那样等于把工具选择权完全交给模型）。
INTENT_TOOL_MAP: dict[Intent, str] = {
    Intent.PRODUCT_QUERY: "product_query",
    Intent.ORDER_QUERY: "order_query",
    Intent.LOGISTICS_QUERY: "logistics_query",
    Intent.RETURN_ELIGIBILITY: "return_eligibility_check",
    Intent.AFTERSALE_CREATE: "aftersale_create",
}

#: 工具所需的必填参数。缺失时走澄清而不是让工具报错。
TOOL_REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "order_query": ("order_id",),
    "logistics_query": ("order_id",),
    "return_eligibility_check": ("order_id",),
    "aftersale_create": ("order_id",),
}

SLOT_LABELS = {"order_id": "订单号", "product_id": "商品编号", "keyword": "商品名称"}

#: 统一的拒答话术。拒答比编造更可取，但话术要给出下一步建议。
NO_ANSWER_REPLY = "抱歉，知识库中没有找到与您问题相关的信息，建议联系人工客服确认。"


class IntentOutput(BaseModel):
    intent: str = "knowledge_qa"
    confidence: float = 0.0
    slots: dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class AnswerOutput(BaseModel):
    status: str = "answered"
    answer: str = ""
    citations: list[str] = Field(default_factory=list)


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        session_store: SessionStore,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = session_store
        self.llm = llm or build_llm_client(settings)
        self.embedder = embedder or build_embedder(settings)
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=settings.chroma_dir, collection_name=settings.collection_name
        )
        self.registry = registry or build_registry()
        self.indexer = KnowledgeIndexer(settings=settings, store=self.vector_store, embedder=self.embedder)
        self._bm25: BM25Index | None = None
        self._graph = build_agent_graph(self)

    # ---------------- 知识库 ----------------

    async def reindex(self, *, mode: str = "incremental") -> IndexReport:
        report = await self.indexer.build(mode=mode)  # type: ignore[arg-type]
        self._bm25 = self.indexer.build_bm25()
        return report

    def knowledge_status(self) -> dict[str, Any]:
        versions = self.vector_store.doc_versions()
        return {
            "collection": self.settings.collection_name,
            "chunk_count": self.vector_store.count(),
            "doc_count": len(versions),
            "doc_versions": versions,
            "bm25_ready": self._bm25 is not None,
            "bm25_size": self._bm25.size if self._bm25 else 0,
            "embedding_provider": self.settings.embedding_provider,
            "retrieval_mode": self.settings.retrieval_mode,
        }

    def _retriever(self, mode: str | None = None) -> HybridRetriever:
        if self._bm25 is None:
            self._bm25 = self.indexer.build_bm25()
        return HybridRetriever(
            settings=self.settings,
            vector=VectorRetriever(self.vector_store, self.embedder),
            bm25=BM25Retriever(self._bm25),
            reranker=LexicalReranker(),
        )

    async def retrieve_only(
        self, query: str, *, mode: str | None = None, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """只做检索不生成回答。评测脚本用它比较不同检索模式。"""
        return await self._retriever().retrieve(query, mode=mode, top_k=top_k)

    # ---------------- 对话 ----------------

    async def chat(
        self,
        *,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        authenticated: bool = False,
    ) -> ChatAnswer:
        started = time.perf_counter()
        trace_id = set_trace_id(get_trace_id() if get_trace_id() != "-" else new_trace_id())
        session_id = session_id or f"s_{uuid.uuid4().hex[:12]}"
        set_session_id(session_id)

        await self.sessions.ensure_session(session_id, user_id)
        history = await self.sessions.history(session_id, user_id)

        state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "authenticated": authenticated,
            "question": question.strip(),
            "history": [{"role": t.role, "content": t.content} for t in history[-6:]],
            "tool_calls": [],
            "retrieved": [],
            "citations": [],
            "dropped_citations": [],
            "needs_human": False,
        }

        try:
            final: AgentState = await self._graph.ainvoke(state)
        except KnowledgeBaseEmptyError:
            raise
        except SMAError as exc:
            logger.warning("chat_failed", extra={"code": exc.code, "error": exc.message})
            final = {
                **state,
                "status": AnswerStatus.ERROR,
                "answer": "抱歉，服务暂时不可用，请稍后重试或联系人工客服。",
            }
        except Exception as exc:  # noqa: BLE001 - 不让内部异常击穿到用户
            logger.exception("chat_crashed")
            final = {
                **state,
                "status": AnswerStatus.ERROR,
                "answer": "抱歉，处理您的问题时出现异常，已记录并将尽快修复。",
            }
            _ = exc

        answer = ChatAnswer(
            session_id=session_id,
            trace_id=trace_id,
            status=final.get("status", AnswerStatus.ERROR),
            answer=final.get("answer", ""),
            intent=(final.get("intent") or IntentResult(intent=Intent.KNOWLEDGE_QA)).intent,
            citations=final.get("citations", []),
            dropped_citations=final.get("dropped_citations", []),
            tool_calls=final.get("tool_calls", []),
            retrieval_trace=final.get("retrieval_trace"),
            needs_human=final.get("needs_human", False),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

        await self.sessions.append_turn(session_id, ChatTurn(role="user", content=question.strip()))
        await self.sessions.append_turn(session_id, ChatTurn(role="assistant", content=answer.answer))
        logger.info(
            "chat_done",
            extra={
                "intent": answer.intent.value,
                "status": answer.status.value,
                "citations": len(answer.citations),
                "tools": len(answer.tool_calls),
                "elapsed_ms": answer.elapsed_ms,
            },
        )
        return answer

    # ---------------- 图节点实现 ----------------

    async def do_detect_intent(self, state: AgentState) -> IntentResult:
        question = state["question"]

        # 确定性注入检测先于模型：模型可能被绕过，模式匹配不会。
        # 命中即短路为 out_of_scope，根本不进入检索和工具分支。
        if pattern := detect_injection(question):
            return IntentResult(
                intent=Intent.OUT_OF_SCOPE,
                confidence=1.0,
                reason=f"命中注入模式: {pattern}",
            )
        # 身份冒充只告警不拦截：真正的保护是鉴权永远用会话里的 user_id
        detect_identity_spoof(question)

        history_text = "\n".join(f"{h['role']}: {h['content']}" for h in state.get("history", []))
        messages = [
            LLMMessage(role="system", content=INTENT_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(f"对话历史：\n{history_text}\n\n" if history_text else "") + f"用户问题：{state['question']}",
            ),
        ]
        try:
            output = await self.llm.complete_json(messages, IntentOutput, purpose="intent")
        except SMAError as exc:
            # 意图识别失败时退化为知识库问答：这是最安全的兜底
            # （不会触发任何工具调用和写操作）
            logger.warning("intent_fallback", extra={"code": exc.code})
            return IntentResult(
                intent=Intent.KNOWLEDGE_QA, confidence=0.0, reason=f"意图识别失败，降级为知识库检索: {exc.code}"
            )

        try:
            intent = Intent(output.intent)
        except ValueError:
            logger.warning("intent_unknown", extra={"raw": output.intent})
            intent = Intent.KNOWLEDGE_QA

        # 安全：模型给出的 slots 里绝不接受身份字段
        slots = {k: v for k, v in output.slots.items() if k not in {"user_id", "uid", "account"}}
        return IntentResult(intent=intent, confidence=output.confidence, slots=slots, reason=output.reason)

    async def do_retrieve(self, state: AgentState) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        return await self._retriever().retrieve(state["question"])

    async def do_answer_from_knowledge(self, state: AgentState) -> dict[str, Any]:
        retrieved: list[RetrievedChunk] = state.get("retrieved", [])
        if not retrieved:
            return {
                "status": AnswerStatus.NO_ANSWER,
                "answer": NO_ANSWER_REPLY,
                "citations": [],
            }

        # 相关性门槛：检索永远会返回 top_k 个结果，哪怕全不相关。
        # 不设门槛就把无关内容喂给模型，它会"基于上下文"编一个答案出来。
        relevant, coverage = relevance_guard(state["question"], retrieved)
        if not relevant:
            logger.info("no_answer_by_guard", extra={"coverage": coverage})
            return {
                "status": AnswerStatus.NO_ANSWER,
                "answer": NO_ANSWER_REPLY,
                "citations": [],
                "raw_citations": [],
            }

        context, used_ids = build_context(retrieved, max_chars=self.settings.context_max_chars)
        if trace := state.get("retrieval_trace"):
            trace.final_chunk_ids = used_ids
            trace.context_chars = len(context)

        messages = [
            LLMMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"用户问题：{state['question']}\n\n【知识库片段】\n{context}",
            ),
        ]
        try:
            output = await self.llm.complete_json(messages, AnswerOutput, purpose="answer")
        except SMAError as exc:
            logger.warning("answer_generation_failed", extra={"code": exc.code})
            return {
                "status": AnswerStatus.ERROR,
                "answer": "抱歉，生成回答时出现问题，请稍后重试或联系人工客服。",
                "citations": [],
            }

        status = AnswerStatus.NO_ANSWER if output.status == "no_answer" else AnswerStatus.ANSWERED
        return {"status": status, "answer": output.answer, "raw_citations": output.citations}

    async def do_call_tool(self, state: AgentState) -> dict[str, Any]:
        intent_result = state["intent"]
        tool_name = INTENT_TOOL_MAP.get(intent_result.intent)
        if tool_name is None:
            return {"status": AnswerStatus.ESCALATED, "answer": OUT_OF_SCOPE_ANSWER, "needs_human": True}

        tool = self.registry.get(tool_name)
        if tool is None:
            logger.error("tool_not_registered", extra={"tool": tool_name})
            return {
                "status": AnswerStatus.ERROR,
                "answer": "抱歉，该功能暂时不可用，请联系人工客服。",
            }

        # 必填参数缺失 → 澄清（不让工具白跑一次再报错）
        missing = [s for s in TOOL_REQUIRED_SLOTS.get(tool_name, ()) if not intent_result.slots.get(s)]
        if missing:
            labels = "、".join(SLOT_LABELS.get(m, m) for m in missing)
            return {
                "status": AnswerStatus.CLARIFY,
                "answer": CLARIFY_TEMPLATE.format(missing=labels),
                "tool_calls": state.get("tool_calls", []),
            }

        ctx = ToolContext(
            user_id=state.get("user_id"),
            session_id=state["session_id"],
            authenticated=bool(state.get("authenticated")),
        )
        result, record = await tool.execute(dict(intent_result.slots), ctx, timeout=self.settings.tool_timeout_seconds)
        calls: list[ToolCallRecord] = [*state.get("tool_calls", []), record]

        if record.status == ToolStatus.FORBIDDEN:
            return {
                "tool_calls": calls,
                "status": AnswerStatus.FORBIDDEN,
                "answer": result.error_message or "您无权访问该资源。",
            }
        if record.status == ToolStatus.NEEDS_APPROVAL:
            return {
                "tool_calls": calls,
                "status": AnswerStatus.ESCALATED,
                "answer": HIGH_RISK_ANSWER,
                "needs_human": True,
            }
        return {
            "tool_calls": calls,
            "tool_data": result.data,
            "tool_summary": result.summary,
            "tool_ok": result.ok,
            "tool_error": result.error_message,
        }

    async def do_answer_from_tool(self, state: AgentState) -> dict[str, Any]:
        # 上一个节点已经给出终态（越权/高风险/澄清）时直接透传
        if state.get("status") in {
            AnswerStatus.FORBIDDEN,
            AnswerStatus.ESCALATED,
            AnswerStatus.CLARIFY,
            AnswerStatus.ERROR,
        }:
            return {}

        summary = state.get("tool_summary", "")  # type: ignore[call-overload]
        tool_ok = state.get("tool_ok", False)  # type: ignore[call-overload]
        if not tool_ok:
            message = state.get("tool_error") or "查询未能成功完成"  # type: ignore[call-overload]
            return {
                "status": AnswerStatus.NO_ANSWER,
                "answer": f"{message}。如需进一步协助，请联系人工客服。",
            }

        messages = [
            LLMMessage(role="system", content=TOOL_ANSWER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"用户问题：{state['question']}\n\n工具返回结果：\n{summary}",
            ),
        ]
        try:
            response = await self.llm.complete(messages, purpose="tool_answer")
            text = response.content.strip()
        except (SMAError, LLMError) as exc:
            # 模型不可用时直接用工具摘要回答：数据是确定性的，不依赖模型也能给出答案
            logger.warning("tool_answer_fallback", extra={"code": getattr(exc, "code", "unknown")})
            text = summary

        return {"status": AnswerStatus.ANSWERED, "answer": text or summary}

    def do_clarify(self, state: AgentState) -> dict[str, Any]:
        intent_result = state["intent"]
        missing = "订单号" if "order_id" not in intent_result.slots else "更具体的信息"
        return {"status": AnswerStatus.CLARIFY, "answer": CLARIFY_TEMPLATE.format(missing=missing)}

    def do_escalate(self, state: AgentState) -> dict[str, Any]:
        intent_result = state["intent"]
        if intent_result.intent == Intent.HIGH_RISK:
            return {"status": AnswerStatus.ESCALATED, "answer": HIGH_RISK_ANSWER, "needs_human": True}
        return {"status": AnswerStatus.ESCALATED, "answer": OUT_OF_SCOPE_ANSWER, "needs_human": False}

    def do_validate(self, state: AgentState) -> dict[str, Any]:
        """引用校验。工具分支没有知识库引用，跳过。"""
        raw_citations: list[str] = state.get("raw_citations", [])  # type: ignore[call-overload]
        retrieved: list[RetrievedChunk] = state.get("retrieved", [])
        status: AnswerStatus = state.get("status", AnswerStatus.ANSWERED)

        if not retrieved:
            return {}

        new_status, citations, dropped = validate_citations(
            state.get("answer", ""), raw_citations, retrieved, status=status
        )
        out: dict[str, Any] = {"status": new_status, "citations": citations, "dropped_citations": dropped}
        if new_status == AnswerStatus.NO_ANSWER and status == AnswerStatus.ANSWERED:
            out["answer"] = "抱歉，我无法从知识库中找到可靠依据来回答这个问题，建议联系人工客服确认。"
        else:
            find_unsupported_numbers(state.get("answer", ""), retrieved)
        return out

    # ---------------- 生命周期 ----------------

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.embedder.aclose()
