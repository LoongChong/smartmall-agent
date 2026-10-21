"""BM25 检索器（中文分词 + 纯 Python 实现）。

为什么不用 rank_bm25：它不支持中文分词，而且 BM25 本身就几十行，
自己实现可以把分词、停用词、参数都暴露出来讲清楚——这正是面试要问的部分。

分词用 jieba。中文电商问题里同义词和口语化表达很多（"啥时候发货" vs "发货时间"），
jieba 的切分 + BM25 的词频加权在这类场景上通常比纯向量更稳。
"""

from __future__ import annotations

import math
from collections import Counter

import jieba

from ..logging_setup import get_logger
from ..models import Chunk

# jieba 默认把词典加载进度打到 stdout，会污染 CLI 的 JSON 输出
jieba.setLogLevel("WARNING")

logger = get_logger(__name__)

# 中文高频虚词 + 电商场景里区分度极低的词
STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "那",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "怎么",
    "什么",
    "可以",
    "能",
    "请问",
    "一下",
    "请",
    " ",
    "\n",
    "\t",
    "，",
    "。",
    "？",
    "！",
    "、",
    "；",
    "：",
    "（",
    "）",
    "《",
    "》",
}


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词 + 停用词过滤 + 小写化。"""
    return [
        token.lower()
        for token in jieba.lcut(text)
        if token.strip() and token not in STOPWORDS and len(token.strip()) > 0
    ]


class BM25Index:
    """标准 BM25（Okapi），k1=1.5、b=0.75。

    参数含义：
    - ``k1`` 控制词频饱和速度：越大，重复出现的词贡献越高。1.2–2.0 是常用区间。
    - ``b`` 控制文档长度归一化强度：1 表示完全归一化，0 表示不考虑长度。
      我们的块长度相对均匀（都在 chunk_size 附近），0.75 的默认值合适。
    """

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens: list[list[str]] = [tokenize(c.content) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs: list[Counter[str]] = [Counter(t) for t in self.doc_tokens]

        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(chunks)
        # 带 +0.5 平滑的 IDF，避免高频词出现负值造成排序错乱
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        logger.info("bm25_built", extra={"chunks": n, "vocab": len(self.idf), "avg_len": round(self.avg_len, 1)})

    def search(self, query: str, *, top_k: int) -> list[tuple[Chunk, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scores: list[tuple[int, float]] = []
        for idx, freqs in enumerate(self.term_freqs):
            score = 0.0
            length = self.doc_len[idx] or 1
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: (-x[1], self.chunks[x[0]].chunk_id))
        return [(self.chunks[i], s) for i, s in scores[:top_k]]

    @property
    def size(self) -> int:
        return len(self.chunks)
