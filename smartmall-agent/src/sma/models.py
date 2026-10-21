"""领域模型：文档、检索结果、意图、工具、会话、回答。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


# ====================== 知识库 ======================


class DocCategory(StrEnum):
    PRODUCT_SPEC = "product_spec"  # 商品参数
    AFTERSALE_POLICY = "aftersale_policy"  # 售后政策
    PAYMENT_FAQ = "payment_faq"  # 支付 FAQ
    GENERAL_FAQ = "general_faq"  # 常见问题


class DocumentMeta(BaseModel):
    """文档元数据。``version`` 用于知识库版本管理与增量更新判断。"""

    doc_id: str
    title: str
    category: DocCategory
    source_path: str = ""
    version: str = Field(default="", description="内容哈希，内容未变则版本不变")
    product_id: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)

    @staticmethod
    def version_of(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class Chunk(BaseModel):
    """切块。``chunk_id`` 由 doc_id + 序号派生，重复构建时稳定，保证索引幂等。"""

    chunk_id: str
    doc_id: str
    content: str
    seq: int
    title: str = ""
    category: DocCategory = DocCategory.GENERAL_FAQ
    product_id: str | None = None
    version: str = ""

    def to_metadata(self) -> dict[str, str | int]:
        meta: dict[str, str | int] = {
            "doc_id": self.doc_id,
            "seq": self.seq,
            "title": self.title,
            "category": self.category.value,
            "version": self.version,
        }
        if self.product_id:
            meta["product_id"] = self.product_id
        return meta


class RetrievedChunk(BaseModel):
    """检索结果。``score`` 的语义随检索器不同而不同，``rank`` 才是跨检索器可比的。"""

    chunk: Chunk
    score: float
    rank: int = 0
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class RetrievalTrace(BaseModel):
    """检索过程留痕：候选、分数、来源、最终上下文。用于排查"为什么答成这样"。"""

    query: str
    mode: str
    vector_candidates: list[str] = Field(default_factory=list)
    bm25_candidates: list[str] = Field(default_factory=list)
    fused_order: list[str] = Field(default_factory=list)
    reranked_order: list[str] = Field(default_factory=list)
    final_chunk_ids: list[str] = Field(default_factory=list)
    context_chars: int = 0


# ====================== 意图与工具 ======================


class Intent(StrEnum):
    KNOWLEDGE_QA = "knowledge_qa"  # 走知识库检索
    ORDER_QUERY = "order_query"  # 订单查询
    LOGISTICS_QUERY = "logistics_query"  # 物流查询
    PRODUCT_QUERY = "product_query"  # 商品结构化查询
    RETURN_ELIGIBILITY = "return_eligibility"  # 退货资格预检查
    AFTERSALE_CREATE = "aftersale_create"  # 创建售后申请（写操作）
    HIGH_RISK = "high_risk"  # 退款/取消订单等，需人工
    CLARIFY = "clarify"  # 信息不足，需澄清
    OUT_OF_SCOPE = "out_of_scope"  # 超出服务范围


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    slots: dict[str, str] = Field(default_factory=dict, description="抽取的参数，如 order_id")
    reason: str = ""


class ToolStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    FORBIDDEN = "forbidden"
    INVALID_ARGS = "invalid_args"
    TIMEOUT = "timeout"
    NEEDS_APPROVAL = "needs_approval"


class ToolCallRecord(BaseModel):
    tool: str
    status: ToolStatus
    args: dict[str, str] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    elapsed_ms: int = 0
    result_summary: str = ""


# ====================== 会话与回答 ======================


class ChatTurn(BaseModel):
    role: str  # user | assistant
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    category: DocCategory
    snippet: str = ""


class AnswerStatus(StrEnum):
    ANSWERED = "answered"  # 有依据的回答
    NO_ANSWER = "no_answer"  # 知识库中没有，明确拒答
    CLARIFY = "clarify"  # 需要用户补充信息
    FORBIDDEN = "forbidden"  # 越权
    ESCALATED = "escalated"  # 需人工介入
    ERROR = "error"


class ChatAnswer(BaseModel):
    session_id: str
    trace_id: str = ""
    status: AnswerStatus
    answer: str
    intent: Intent = Intent.KNOWLEDGE_QA
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    needs_human: bool = False
    elapsed_ms: int = 0


# ====================== 业务实体（模拟数据）======================


class OrderStatus(StrEnum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDING = "refunding"


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    stock: int
    attributes: dict[str, str] = Field(default_factory=dict)
    return_days: int = 7


class Order(BaseModel):
    order_id: str
    user_id: str
    product_id: str
    product_name: str
    quantity: int
    amount: float
    status: OrderStatus
    created_at: datetime
    delivered_at: datetime | None = None
    tracking_no: str | None = None


class LogisticsEvent(BaseModel):
    time: datetime
    location: str
    description: str


class Logistics(BaseModel):
    tracking_no: str
    carrier: str
    status: str
    events: list[LogisticsEvent] = Field(default_factory=list)


class ReturnEligibility(BaseModel):
    """退货资格判定结果。``eligible`` 由**代码规则**决定，不由模型决定。"""

    order_id: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    policy_days: int = 7
    days_since_delivery: int | None = None
    deadline: datetime | None = None


class AftersaleTicket(BaseModel):
    ticket_id: str
    order_id: str
    user_id: str
    reason: str
    status: str = "submitted"
    created_at: datetime = Field(default_factory=utcnow)
