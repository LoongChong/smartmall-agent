# 检索：三种检索器与 RRF 融合

## 1. 四种模式

`SMA_RETRIEVAL_MODE` 可选四个值，都能跑评测对比：

| 模式 | 组成 | 适合 |
|------|------|------|
| `vector` | 向量单路 | 语义相似（需真实 embedding 才有意义） |
| `bm25` | 词法单路 | 精确术语、型号、专有名词 |
| `hybrid`（默认） | 向量 + BM25 → RRF | 两者互补，综合最稳 |
| `hybrid_rerank` | hybrid → LexicalReranker | 需要更高 Precision 的场景 |

## 2. 向量检索

```python
vector = await embedder.embed_one(query)
hits = store.query(vector, top_k=candidate_k, category=category)
```

两个设计点：

**索引为空时抛错，不返回空列表。** `KnowledgeBaseEmptyError` 带 HTTP 503。
如果静默返回空，用户看到的是"没找到答案"，而真实原因是"忘了执行 `sma index`"——
这两件事的修法完全不同，不能混为一谈。

**category 过滤下推到 Chroma**（`where={"category": ...}`），由向量库在检索时过滤，
而不是先取回 top_k 再在 Python 里筛——后者会导致过滤后结果不足 top_k。

## 3. BM25 检索

### 3.1 为什么自己实现

没有用 `rank_bm25`：它不处理中文分词，而 BM25 本体就几十行。自己写能把
分词器、停用词表、`k1`/`b` 参数全部暴露出来——这些恰好是需要讲清楚的部分。

### 3.2 分词

```python
jieba.setLogLevel("WARNING")  # 默认把词典加载进度打到 stdout，会污染 CLI 的 JSON 输出


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut(text) if t.strip() and t not in STOPWORDS]
```

jieba 精确模式 + 停用词过滤。停用词表包含两类：中文高频虚词（的/了/是/在）
和**电商场景里区分度极低的词**（请问、一下、怎么、可以）。

后一类很关键。"请问一下怎么退货"分词后是 `请问/一下/怎么/退货`，
如果不过滤前三个词，它们会在每条用户问题里出现，变成噪声词——
而且因为它们在文档里几乎不出现，IDF 会很高，反而抢走权重。

### 3.3 打分公式

标准 Okapi BM25，`k1=1.5`、`b=0.75`：

```
score(q, d) = Σ_{t∈q}  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

参数含义：

- **`k1` 控制词频饱和速度**。词出现 10 次不应该比出现 1 次得分高 10 倍
  （边际收益递减）。`k1` 越大饱和越慢，1.2–2.0 是常用区间。
- **`b` 控制文档长度归一化强度**。`b=1` 完全按长度归一化（长文档被重罚），
  `b=0` 完全不考虑长度。我们的块长度相对均匀（都在 `chunk_size=320` 附近），
  0.75 的默认值合适；如果块长差异大，应该调高 `b`。

**IDF 用了 `log(1 + ...)` 形式而不是裸 `log((N-df+0.5)/(df+0.5))`。**
后者在 `df > N/2` 时会变成负数——即"在超过一半文档里出现的词，命中它反而扣分"。
6 篇文档、28 个块的小语料上这种情况很常见（"退货"这个词可能出现在一半的块里），
负 IDF 会把排序彻底搅乱。加 1 保证 IDF 恒正。

### 3.4 排序的确定性

```python
scores.sort(key=lambda x: (-x[1], self.chunks[x[0]].chunk_id))
```

分数相同时按 chunk_id 排——**不加这个 tie-break，同样的输入会得到不同的顺序**，
评测结果就不可复现了。小语料上分数相同的情况很常见。

## 4. RRF 融合

### 4.1 为什么不是加权相加

直觉做法：`score = α · vector_score + (1-α) · bm25_score`。

问题在量纲：向量相似度是 0–1 的余弦，BM25 分数是**无上界的实数**
（取决于 IDF 和词频，可能是 2.3 也可能是 47）。直接相加等于让 BM25 单方面主导。

那就先归一化？归一化方式（min-max / z-score）都依赖**当前这批候选的分布**。
同一个块在不同 query 下，因为同批候选的最高分不同，归一化后的分数会不同——
相对顺序随查询漂移，不稳定，也没法调参。

### 4.2 RRF 只用排名

```
score(d) = Σ_r  1 / (k + rank_r(d))        k = 60
```

```python
for ranking in rankings:
    for item in ranking:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item.rank)
```

三个好处：

1. **跨量纲天然可比**。排名是序数，不关心原始分数的尺度。
2. **对单路异常高分不敏感**。某一路给了某块一个离谱的高分，
   它在 RRF 里也只是"排名第 1"，贡献 `1/61`，不会压倒一切。
3. **无需调权重**。没有 `α` 要调，少一个需要评测集才能确定的超参。

`k=60` 是 RRF 原论文的常用取值。它的作用是**压缩头部差距**：
排名 1 贡献 `1/61 ≈ 0.0164`，排名 2 贡献 `1/62 ≈ 0.0161`，差距只有 2%。
这意味着"两路都排在前几名"比"一路排第 1、另一路没召回"更有优势——
正是融合想要的效果。`k` 调小会让第一名的权重陡增，退化成"谁家第一名说话算"。

代价要讲清楚：**RRF 丢弃了分数里的置信度信息**。
"相似度 0.95 的第一名"和"相似度 0.31 的第一名"在 RRF 眼里一样。
所以 `min_score` 过滤（丢弃低分候选）和相关性门槛（`guards.relevance_guard`）
必须单独补上这一块，它们不是冗余设计。

### 4.3 融合实现的两个细节

```python
if cid not in best or item.rank < best[cid].rank:
    best[cid] = item  # 保留排名更靠前的那条作为代表
...
retriever = "+".join(sorted(set(sources[cid])))  # "bm25+vector"，记录来源
ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))  # 同分按 chunk_id
```

- 同一个块被两路都召回时，保留排名靠前的那条作为展示代表（原始分数更有意义）；
- `retriever` 字段记录来源，trace 里能看到某块是"两路都召回"还是"只有 BM25 召回"——
  这个信息在排查检索问题时很有用；
- 同样有 tie-break，保证确定性。

## 5. 召回宽度与截断

```python
candidate_k = max(top_k, settings.candidate_k)  # 默认 max(4, 10) = 10
```

每路召回 10 个，融合后才截断到 `top_k=4`。

**召回阶段要宽，排序阶段才收窄。** 如果每路只召回 4 个，融合时的信息量就只有 8 条
（且很可能高度重叠），RRF 的互补效应发挥不出来。把 `candidate_k` 开大的成本只是
BM25 多算几条和一点融合开销（实测 p50 < 1ms），收益是融合质量。

最后一道过滤：

```python
final = [h for h in fused if h.score >= settings.min_score][:top_k]
```

`min_score` 默认 0.0（不过滤）。注意 RRF 分数的量级很小（0.016 左右），
如果要用这个阈值，**必须按当前模式的实际分数量级设**，
把向量模式下的阈值（0–1 量级）直接拿来用在 hybrid 模式上会把所有候选全过滤掉。

## 6. 实测对比

> 环境：macOS-27.0.1-arm64，Python 3.11.16，`embedding_provider=hash`，
> `chunk_size=320`，`rrf_k=60`，12 条检索标注用例。
> 原始数据：[`docs/benchmarks/sweep_2026-10-16.json`](benchmarks/sweep_2026-10-16.json)

| K | 模式 | Recall@K | MRR | Precision@K | p50 延迟 |
|---|------|----------|-----|-------------|---------|
| 1 | vector | **0.9583** | **1.0** | **1.0** | 0.58 ms |
| 1 | bm25 | 0.8750 | 0.9167 | 0.9167 | 0.03 ms |
| 1 | hybrid | **0.9583** | **1.0** | **1.0** | 0.59 ms |
| 1 | hybrid_rerank | 0.8750 | 0.9167 | 0.9167 | 1.60 ms |
| 2 | vector | 1.0 | **1.0** | 0.7500 | 0.53 ms |
| 2 | bm25 | 1.0 | 0.9583 | 0.8333 | 0.02 ms |
| 2 | hybrid | 1.0 | **1.0** | 0.7917 | 0.57 ms |
| 2 | hybrid_rerank | 1.0 | 0.9583 | **0.9167** | 1.60 ms |
| 4 | vector | 1.0 | **1.0** | 0.5069 | 0.55 ms |
| 4 | bm25 | 1.0 | 0.9583 | 0.7083 | 0.02 ms |
| 4 | hybrid | 1.0 | **1.0** | 0.6250 | 0.57 ms |
| 4 | hybrid_rerank | 1.0 | 0.9583 | **0.7639** | 1.57 ms |

读这张表必须带上三条前提，否则会得出错误结论：

1. **K=2 起 Recall 全为 1.0 是天花板效应**，不是"方案完美"。
   只有 6 篇文档，命中正确文档太容易。区分度只体现在 K=1 的 Recall 和各 K 的 Precision 上。
2. **`vector` 这一列是哈希向量的表现，不是语义向量的表现。**
   hash embedding 做词面匹配，和 BM25 的能力高度重叠，所以 hybrid 相比单路
   提升有限——真实 embedding 下两路互补性会强得多，这张表的相对排名会变。
3. **延迟数字只能横向比较，不能当生产指标。** 全部是本地内存操作，
   没有网络调用、没有真实 embedding 的 API 往返（那是几十到几百毫秒量级）。

从这张表能可靠得出的结论只有两条：
BM25 快一个数量级（0.02ms vs 0.55ms，向量路要算 query embedding）；
重排在 K≥2 时稳定提升 Precision 但略微损害 MRR——详见 [RERANKING](RERANKING.md)。

## 7. 复现

```bash
uv run sma index
uv run sma eval                              # K=4 的四模式对比
uv run python scripts/eval_sweep.py          # K=1..4 扫描，输出到 docs/benchmarks/
```

换模式只需改环境变量，索引不用重建（四种模式共用同一份索引）：

```bash
SMA_RETRIEVAL_MODE=bm25 uv run sma ask "星云 Pro 14 的屏幕刷新率是多少"
```
