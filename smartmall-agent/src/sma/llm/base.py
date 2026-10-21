"""LLM 客户端抽象。

对上层只暴露两个能力：
- ``complete``：自由文本补全；
- ``complete_json``：结构化输出，内部负责 JSON 提取与 Pydantic 校验。

把"结构化"收敛到这一层，Planner / Synthesizer 就不必各自写一遍解析和容错。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import LLMOutputFormatError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(slots=True)
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int = 0


def extract_json(text: str) -> Any:
    """从模型输出里抽取 JSON。

    依次尝试：整体解析 → 去掉 ```json 围栏 → 截取第一个 ``{`` 到最后一个 ``}``。
    三种都失败才抛 ``LLMOutputFormatError``。
    """
    candidates: list[str] = [text.strip()]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMOutputFormatError("模型输出无法解析为 JSON", detail=text[:500])


class LLMClient(ABC):
    """所有模型提供方的统一接口。"""

    name: str = "base"

    @abstractmethod
    async def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse: ...

    async def complete_json(self, messages: list[LLMMessage], schema: type[T], **kwargs: Any) -> T:
        response = await self.complete(messages, response_format={"type": "json_object"}, **kwargs)
        data = extract_json(response.content)
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputFormatError(f"模型输出不满足 {schema.__name__} 约束", detail=str(exc)[:800]) from exc

    async def aclose(self) -> None:  # pragma: no cover - 默认无资源
        return None
