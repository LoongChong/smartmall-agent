"""切块策略。

针对中文电商文档的特点做了两处设计：

1. **按 Markdown 小标题优先切分**。售后政策这类文档里，"不支持无理由退货的情况"
   和"退货运费"属于不同语义单元，按固定长度硬切会把它们混在一个块里，
   检索时召回了也答不准。标题切分天然对齐语义边界。
2. **超长小节再按句号二次切分，并带重叠**。单个小节超过 chunk_size 时，
   按中文句末标点切句，累积到阈值成块，块间保留 overlap 字符，
   避免答案正好落在切点上被劈成两半。

块的标题会被拼进正文（`标题\n内容`），因为 BM25 和向量都受益于标题中的关键词。
"""

from __future__ import annotations

import re

from ..models import Chunk, DocumentMeta

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[。！？；\n])")


def split_sections(body: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切成 (小节标题, 小节正文)。标题前的内容归入 "" 标题。"""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    buffer: list[str] = []
    for line in body.splitlines():
        if match := _HEADING_RE.match(line):
            if buffer and any(s.strip() for s in buffer):
                sections.append((current_title, buffer))
            current_title = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer and any(s.strip() for s in buffer):
        sections.append((current_title, buffer))
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def split_long_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """按句切分并累积到 chunk_size，块间保留 overlap 字符。"""
    if len(text) <= chunk_size:
        return [text]

    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) > chunk_size:
            chunks.append(buffer.strip())
            buffer = (buffer[-overlap:] if overlap else "") + sentence
        else:
            buffer += sentence
        # 单句本身就超长（少见，如一整段没有标点的表格）
        while len(buffer) > chunk_size * 2:
            chunks.append(buffer[:chunk_size].strip())
            buffer = buffer[chunk_size - overlap :]
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def chunk_document(meta: DocumentMeta, body: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """把一篇文档切成 Chunk 列表。chunk_id 稳定可重建，保证索引幂等。"""
    pieces: list[str] = []
    for section_title, section_body in split_sections(body):
        prefix = f"{meta.title} - {section_title}\n" if section_title else f"{meta.title}\n"
        for part in split_long_text(section_body, chunk_size=chunk_size, overlap=overlap):
            pieces.append(prefix + part)

    if not pieces:  # 整篇没有标题结构
        pieces = [f"{meta.title}\n{p}" for p in split_long_text(body, chunk_size=chunk_size, overlap=overlap)]

    return [
        Chunk(
            chunk_id=f"{meta.doc_id}#{seq}",
            doc_id=meta.doc_id,
            content=content,
            seq=seq,
            title=meta.title,
            category=meta.category,
            product_id=meta.product_id,
            version=meta.version,
        )
        for seq, content in enumerate(pieces)
    ]
