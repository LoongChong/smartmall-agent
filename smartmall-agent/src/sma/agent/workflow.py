"""Agent 工作流（LangGraph）。

图结构：

    route ──┬─→ retrieve → answer   ──→ validate → END
            ├─→ tool ───→ tool_answer ─→ validate → END
            ├─→ clarify ─────────────────────────→ END
            └─→ escalate ────────────────────────→ END

为什么这里用了 LangGraph，而 Deep Research Agent 的子任务调度没用：
这个流程的核心就是**条件路由**——根据意图走完全不同的分支，
且分支数量固定、运行期不变。这正是图编排的适用场景。
而子任务调度需要动态建图 + 并发控制，不适合。

有限循环：图中没有回边，每条路径最多经过 4 个节点就到 END。
工具调用次数由 ``max_tool_calls`` 限制（当前实现每轮最多调 1 个工具，上限是为将来多工具编排预留）。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..logging_setup import get_logger
from ..models import (
    AnswerStatus,
    Citation,
    Intent,
    IntentResult,
    RetrievalTrace,
    RetrievedChunk,
    ToolCallRecord,
)

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """图状态。

    只放单轮对话推进所需的数据；会话历史与业务数据分别在 SessionStore 和工具层，
    不在这里重复持有。
    """

    session_id: str
    user_id: str | None
    authenticated: bool
    question: str
    history: list[dict[str, str]]

    intent: IntentResult
    retrieved: list[RetrievedChunk]
    retrieval_trace: RetrievalTrace | None

    tool_calls: list[ToolCallRecord]
    tool_data: dict[str, Any] | None
    tool_summary: str
    tool_ok: bool
    tool_error: str | None

    answer: str
    # 模型原始输出的引用，未经校验。validate 节点会把它换成真实的 citations。
    # 注意：LangGraph 会丢弃未在 TypedDict 中声明的键，所有跨节点传递的字段都必须在这里列出。
    raw_citations: list[str]
    status: AnswerStatus
    citations: list[Citation]
    dropped_citations: list[str]
    needs_human: bool


#: 意图 → 分支。集中在一处，便于审计"什么情况下会走到工具/人工"。
INTENT_ROUTES: dict[Intent, str] = {
    Intent.KNOWLEDGE_QA: "retrieve",
    Intent.PRODUCT_QUERY: "tool",
    Intent.ORDER_QUERY: "tool",
    Intent.LOGISTICS_QUERY: "tool",
    Intent.RETURN_ELIGIBILITY: "tool",
    Intent.AFTERSALE_CREATE: "tool",
    Intent.HIGH_RISK: "escalate",
    Intent.CLARIFY: "clarify",
    Intent.OUT_OF_SCOPE: "escalate",
}


def build_agent_graph(service: Any) -> Any:
    """用 ``ChatService`` 的方法装配图。参数用 Any 避免循环导入。"""

    async def route_node(state: AgentState) -> AgentState:
        intent = await service.do_detect_intent(state)
        return {**state, "intent": intent}

    async def retrieve_node(state: AgentState) -> AgentState:
        retrieved, trace = await service.do_retrieve(state)
        return {**state, "retrieved": retrieved, "retrieval_trace": trace}

    async def answer_node(state: AgentState) -> AgentState:
        return {**state, **(await service.do_answer_from_knowledge(state))}

    async def tool_node(state: AgentState) -> AgentState:
        return {**state, **(await service.do_call_tool(state))}

    async def tool_answer_node(state: AgentState) -> AgentState:
        return {**state, **(await service.do_answer_from_tool(state))}

    async def clarify_node(state: AgentState) -> AgentState:
        return {**state, **service.do_clarify(state)}

    async def escalate_node(state: AgentState) -> AgentState:
        return {**state, **service.do_escalate(state)}

    async def validate_node(state: AgentState) -> AgentState:
        return {**state, **service.do_validate(state)}

    def route_by_intent(state: AgentState) -> str:
        intent = state.get("intent")
        target = INTENT_ROUTES.get(intent.intent if intent else Intent.KNOWLEDGE_QA, "retrieve")
        logger.info(
            "intent_routed",
            extra={"intent": intent.intent.value if intent else "?", "node": target},
        )
        return target

    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("tool_answer", tool_answer_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        route_by_intent,
        {"retrieve": "retrieve", "tool": "tool", "clarify": "clarify", "escalate": "escalate"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "validate")
    graph.add_edge("tool", "tool_answer")
    graph.add_edge("tool_answer", "validate")
    graph.add_edge("validate", END)
    graph.add_edge("clarify", END)
    graph.add_edge("escalate", END)
    return graph.compile()
