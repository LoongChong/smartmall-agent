"""确定性 Mock LLM。

用途：
1. 无 API Key 跑通全流程与 CI；
2. 评测需要可复现的回答（真实模型每次输出不同，无法做回归）；
3. 边界场景（JSON 损坏、超时、拒答）需要确定性输入。

它按 ``purpose`` 分路：
- ``intent``：基于关键词的规则意图识别（**刻意不做得太聪明**，
  这样意图识别的边界由测试显式覆盖，而不是靠模型碰运气）；
- ``answer``：从上下文中回读真实 chunk_id 生成带引用的回答；证据为空时返回拒答。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

from ..errors import LLMError
from .base import LLMClient, LLMMessage, LLMResponse

_CHUNK_ID_RE = re.compile(r"\[([A-Za-z0-9_\-]+#\d+)\]")

#: 意图关键词表。顺序即优先级：越靠前的规则越先匹配。
_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("high_risk", ("退款", "退钱", "取消订单", "赔偿", "投诉到")),
    ("aftersale_create", ("申请售后", "提交售后", "申请退货", "我要退货", "发起售后")),
    ("return_eligibility", ("能退吗", "可以退吗", "能不能退", "退货资格", "符合退货", "还能退")),
    ("logistics_query", ("物流", "快递", "到哪了", "什么时候到", "运单", "签收了吗")),
    ("order_query", ("订单", "我买的", "下单", "order", "so2026")),
    ("product_query", ("库存", "有货", "多少钱", "价格", "售价", "参数", "配置")),
]

_ORDER_RE = re.compile(r"\b(SO\d{10,})\b", re.IGNORECASE)
_PRODUCT_RE = re.compile(r"\b(P\d{4})\b", re.IGNORECASE)


class MockLLMClient(LLMClient):
    name = "mock"

    def __init__(
        self,
        *,
        latency_seconds: float = 0.0,
        scripted: Sequence[str] | None = None,
        fail_times: int = 0,
        failure: Exception | None = None,
    ) -> None:
        self.latency_seconds = latency_seconds
        self._scripted = list(scripted) if scripted else None
        self._fail_times = fail_times
        self._failure = failure or LLMError("mock 注入的失败")
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        purpose = kwargs.get("purpose", "generic")
        self.calls.append({"purpose": purpose, "messages": [m.content for m in messages]})
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._failure
        if self._scripted:
            content = self._scripted.pop(0)
        elif purpose == "intent":
            content = json.dumps(self._intent(messages), ensure_ascii=False)
        elif purpose == "answer":
            content = json.dumps(self._answer(messages), ensure_ascii=False)
        elif purpose == "tool_answer":
            # 直接复述工具摘要：工具数据是确定性的，Mock 不应该在这里加工或编造
            content = self._tool_answer(messages)
        else:
            content = "mock-response"
        return LLMResponse(content=content, model="mock-model", prompt_tokens=20, completion_tokens=30)

    # ---------- 分路实现 ----------

    @staticmethod
    def _user_text(messages: Sequence[LLMMessage]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""

    def _intent(self, messages: Sequence[LLMMessage]) -> dict[str, Any]:
        text = self._user_text(messages)
        question = text.split("用户问题：")[-1].strip() if "用户问题：" in text else text
        lowered = question.lower()

        slots: dict[str, str] = {}
        if m := _ORDER_RE.search(question):
            slots["order_id"] = m.group(1).upper()
        if m := _PRODUCT_RE.search(question):
            slots["product_id"] = m.group(1).upper()

        for intent, keywords in _INTENT_RULES:
            if any(k.lower() in lowered for k in keywords):
                # 需要订单号但没给 → 转澄清
                if intent in {"order_query", "logistics_query", "return_eligibility", "aftersale_create"} and (
                    "order_id" not in slots
                ):
                    return {
                        "intent": "clarify",
                        "confidence": 0.6,
                        "slots": slots,
                        "reason": "缺少订单号",
                    }
                return {"intent": intent, "confidence": 0.85, "slots": slots, "reason": f"命中关键词规则 {intent}"}

        return {"intent": "knowledge_qa", "confidence": 0.7, "slots": slots, "reason": "默认走知识库检索"}

    def _tool_answer(self, messages: Sequence[LLMMessage]) -> str:
        text = self._user_text(messages)
        if "工具返回结果：" in text:
            return text.split("工具返回结果：", 1)[1].strip()
        return text.strip()

    def _answer(self, messages: Sequence[LLMMessage]) -> dict[str, Any]:
        blob = "\n".join(m.content for m in messages if m.role == "user")
        chunk_ids = list(dict.fromkeys(_CHUNK_ID_RE.findall(blob)))
        if not chunk_ids:
            return {
                "status": "no_answer",
                "answer": "抱歉，知识库中没有找到与您问题相关的信息，建议联系人工客服确认。",
                "citations": [],
            }
        return {
            "status": "answered",
            "answer": "(mock) 根据知识库资料，已为您整理相关说明。",
            "citations": chunk_ids[:3],
        }
