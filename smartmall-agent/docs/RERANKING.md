# 重排：收益与代价

## 1. 重排要解决的具体问题

RRF 融合只看排名。当两个块的融合分接近时，它无法区分这两种情况：

- 块 A：问题的 5 个实词全部命中；
- 块 B：只命中 1 个词，但恰好在两路召回里排名都还不错。

对 RRF 来说 A 和 B 可能分数相当。对用户来说 A 明显更该排在前面。

重排是流水线的**最后一步**，只对已召回的候选重新排序，不参与召回。
它改变不了"没召回到"的问题——这一点很重要：如果 Recall 不够，加重排是白费力气。

## 2. 当前实现：LexicalReranker

**它不是 cross-encoder，文档里不会把它说成语义重排。**

```python
score = coverage + position_bonus + title_bonus + base
```

四项（`rag/retriever.py:LexicalReranker`）：

| 项 | 计算 | 权重 | 为什么 |
|----|------|------|--------|
| `coverage` | 问题实词 ∩ 块实词 / 问题实词 | 0–1（主项） | 直接解决上面那个问题 |
| `position_bonus` | `(1 - 首次命中位置/块长) × 0.2` | 0–0.2 | 答案通常在块开头，命中词出现在末尾往往只是顺带提及 |
| `title_bonus` | 问题词命中块标题则 +0.15 | 0 / 0.15 | 标题命中是强主题信号 |
| `base` | `1/(10 + 原融合排名)` | 0.03–0.09 | **保留召回阶段的判断，避免重排完全推翻原排序** |

`base` 项值得单独说。没有它，重排完全由词面覆盖率决定，等于把向量检索的贡献
全部丢掉——那还不如直接用 BM25。`1/(10+rank)` 的量级（约 0.03–0.09）
相当于"覆盖率差 5% 以内时，听召回阶段的"。10 这个常数比 RRF 的 60 小得多，
是因为这里候选数量少（10 个），需要更明显的排名区分度。

同样有 tie-break：`scored.sort(key=lambda x: (-x[0], x[1].chunk_id))`。

## 3. 实测收益与代价

> 环境同 [RETRIEVAL](RETRIEVAL.md)：macOS-27.0.1-arm64，Python 3.11.16，
> hash embedding，12 条检索标注用例。
> 原始数据：[`docs/benchmarks/sweep_2026-10-16.json`](benchmarks/sweep_2026-10-16.json)

| K | 指标 | hybrid | hybrid_rerank | 变化 |
|---|------|--------|---------------|------|
| 1 | Recall | 0.9583 | 0.8750 | **−8.3%** |
| 1 | MRR | 1.0 | 0.9167 | **−8.3%** |
| 1 | Precision | 1.0 | 0.9167 | **−8.3%** |
| 2 | Recall | 1.0 | 1.0 | 0 |
| 2 | MRR | 1.0 | 0.9583 | **−4.2%** |
| 2 | Precision | 0.7917 | **0.9167** | **+15.8%** |
| 3 | Precision | 0.6806 | **0.7917** | **+16.3%** |
| 4 | Precision | 0.6250 | **0.7639** | **+22.2%** |
| 4 | MRR | 1.0 | 0.9583 | −4.2% |
| — | p50 延迟 | 0.57 ms | 1.57 ms | **+1.0 ms（约 2.8×）** |

### 这组数字说明什么

**收益是 Precision，而且随 K 增大而增大**（K=2 时 +15.8%，K=4 时 +22.2%）。
符合预期：K 越大，进入上下文的无关块越多，重排把它们压下去的价值就越明显。
Precision 高意味着喂给模型的上下文噪声少，生成质量更稳。

**代价是 MRR 下降 4.2%**。12 条用例里，MRR 从 1.0 降到 0.9583 意味着
**有一条用例的首个正确文档从第 1 位掉到了第 2 位**（`1/2` 对 12 取平均正好是 −0.0417）。

这不是统计噪声，是一个可解释的机制性缺陷：词面覆盖率会**高估"词多但主题偏"的块**。
一个块如果恰好包含了问题里的多个词，但整体在讲另一件事，覆盖率会把它推上来。
cross-encoder 不会犯这个错，因为它看的是语义匹配而不是词面交集。

**K=1 时三个指标全降**。只取 1 个块时，重排把本来排第 1 的正确块换掉了，
没有第 2 名来兜。所以 **`hybrid_rerank` 在 K=1 下是净亏损的**。

### 由此得到的使用建议

| 场景 | 建议 |
|------|------|
| `top_k = 1` | **不要用重排**，用 `hybrid` |
| `top_k ≥ 2`（本项目默认 4） | 用 `hybrid_rerank`，Precision 收益明显大于 MRR 损失 |
| 对首位准确率极敏感（如只展示一条答案） | 用 `hybrid`，或换真正的 cross-encoder |
| 延迟预算极紧 | 评估 +1ms 是否可接受（相对 LLM 的几百毫秒可忽略） |

默认配置是 `SMA_RETRIEVAL_MODE=hybrid` + `SMA_TOP_K=4`。
没有把默认改成 `hybrid_rerank`，原因是：MRR 的损失是**机制性的**（词面重排的固有缺陷），
而 Precision 的收益在当前这个 6 篇文档的小语料上**可能被放大**
（文档少，无关块的词面交集更容易偶然偏高）。默认选更保守的那个，
把 `hybrid_rerank` 作为显式可选项，并把数据摆在这里让使用者自己判断。

## 4. 这些数字的局限（必读）

1. **12 条用例，一条用例的变化就是 8.3%。** 这个样本量下，±1 条用例的
   差异在统计上不显著。表里的百分比只能说明方向，不能当精确的效果量。
2. **hash embedding 下的结论不能直接外推。** 词面哈希向量和词面重排的能力
   高度同源——重排在纠正向量路的错误上，收益可能被低估；
   在和 BM25 的重叠上，收益可能被高估。换真实 embedding 后必须重跑。
3. **6 篇文档的天花板效应。** K≥2 时 Recall 已经是 1.0，重排在 Recall 上
   不可能有任何表现空间。语料扩大后这张表的结构会变。
4. **延迟是纯本地计算。** 1.57ms 里不含任何网络 I/O。接入 cross-encoder 后
   延迟会是几十到几百毫秒量级（取决于是本地推理还是 API），完全不可比。

## 5. 换成真正的 cross-encoder

接口已经留好了：

```python
class Reranker(ABC):
    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]: ...
```

接 `bge-reranker-v2-m3` 的大致形状：

```python
class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query, candidates, *, top_k):
        pairs = [(query, c.chunk.content) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: (-x[0], x[1].chunk_id))
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s), rank=i + 1, retriever=f"{c.retriever}|xenc")
            for i, (s, c) in enumerate(ranked[:top_k])
        ]
```

然后注入：

```python
HybridRetriever(settings=settings, vector=..., bm25=..., reranker=CrossEncoderReranker())
```

换之前要考虑的四件事：

- **延迟**：每个候选都要过一次模型。`candidate_k=10` 意味着 10 次前向推理。
  本地 CPU 上 bge-reranker-v2-m3 大约几百毫秒，必须实测后再决定 `candidate_k`。
- **同步阻塞**：`CrossEncoder.predict` 是同步的，直接在 async 节点里调会阻塞事件循环。
  要包 `asyncio.to_thread`，或者走独立的推理服务。
- **依赖体积**：`sentence-transformers` + torch 会让镜像从几百 MB 涨到几个 GB。
  这是当前没有默认引入它的实际原因之一。
- **评测回归**：换完必须重跑 `scripts/eval_sweep.py`，用同一组用例对比。
  不要凭"cross-encoder 肯定更好"直接上——在短文本、强词面匹配的场景下，
  它未必能赢过 BM25 + 词面重排的组合。

> **本项目没有实际跑过 cross-encoder。** 上面的代码是接口演示，
> 不是经过验证的实现，文档里也没有任何 cross-encoder 的性能数字。
