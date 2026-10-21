"""文档加载与元数据提取。

格式：Markdown + YAML frontmatter（手写极简解析，不引入 PyYAML——
frontmatter 只有扁平的 key: value，正则足够，少一个依赖）。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import IndexingError
from ..logging_setup import get_logger
from ..models import DocCategory, DocumentMeta

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """返回 (元数据字典, 正文)。没有 frontmatter 时返回 ({}, 原文)。"""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if kv := _KV_RE.match(line):
            meta[kv.group(1)] = kv.group(2).strip().strip("\"'")
    return meta, text[match.end() :]


def load_document(path: Path) -> tuple[DocumentMeta, str]:
    """加载单个文档。元数据缺失时用文件名与内容推断，不直接报错。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IndexingError(f"读取文档失败: {path}", detail=str(exc)) from exc

    front, body = parse_frontmatter(raw)
    body = body.strip()
    if not body:
        raise IndexingError(f"文档正文为空: {path}")

    category_raw = front.get("category", "")
    try:
        category = DocCategory(category_raw)
    except ValueError:
        category = _infer_category(path.name)
        logger.warning(
            "doc_category_inferred",
            extra={"path": str(path), "raw": category_raw, "inferred": category.value},
        )

    title = front.get("title") or _first_heading(body) or path.stem
    meta = DocumentMeta(
        doc_id=front.get("doc_id") or path.stem,
        title=title,
        category=category,
        source_path=str(path),
        version=DocumentMeta.version_of(body),
        product_id=front.get("product_id") or None,
    )
    return meta, body


def load_directory(directory: Path) -> list[tuple[DocumentMeta, str]]:
    """加载目录下所有 .md 文档。目录不存在或为空时抛错，不静默返回空列表。"""
    if not directory.exists():
        raise IndexingError(f"知识库目录不存在: {directory}")
    files = sorted(directory.glob("*.md"))
    if not files:
        raise IndexingError(f"知识库目录下没有 .md 文档: {directory}")

    docs: list[tuple[DocumentMeta, str]] = []
    seen_ids: set[str] = set()
    for path in files:
        meta, body = load_document(path)
        if meta.doc_id in seen_ids:
            raise IndexingError(f"重复的 doc_id: {meta.doc_id} ({path})")
        seen_ids.add(meta.doc_id)
        docs.append((meta, body))
    logger.info("docs_loaded", extra={"count": len(docs), "dir": str(directory)})
    return docs


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _infer_category(filename: str) -> DocCategory:
    name = filename.lower()
    if "product" in name or "spec" in name:
        return DocCategory.PRODUCT_SPEC
    if "aftersale" in name or "return" in name or "warranty" in name:
        return DocCategory.AFTERSALE_POLICY
    if "pay" in name or "invoice" in name:
        return DocCategory.PAYMENT_FAQ
    return DocCategory.GENERAL_FAQ
