# 切块策略与向量化

## 1. 为什么不按固定长度硬切

最省事的切块是"每 320 个字一刀"。在中文电商文档上这会立刻出问题。

看 `aftersale_return_policy.md` 的结构：

```markdown
## 退货时限
7 天无理由退货自签收之日起计算…

## 不支持无理由退货的情况
定制类商品、已拆封的个人卫生用品…

## 退货运费
质量问题由平台承担，无理由退货由买家承担…
```

固定长度切分会把"不支持退货的情况"的后半段和"退货运费"的前半段塞进同一个块。
用户问"退货运费谁出"，这个块被召回了——但块里一半内容在讲"哪些商品不能退"，
模型很可能把两件事串起来答。**召回正确 ≠ 回答正确**，噪声内容会污染生成。

## 2. 实际策略：标题优先 + 超长二次切分

`knowledge/chunker.py` 两级切分：

### 第一级：按 Markdown 标题

```python
split_sections(body) -> list[tuple[str, str]]   # [(小节标题, 小节正文), ...]
```

正则 `^(#{1,6})\s+(.*)$` 匹配 1–6 级标题，标题之间的内容归为一节。
标题前的内容（通常是文档导语）归入空标题那一节，不丢弃。

Markdown 标题天然就是作者标注的语义边界——**这是免费的、人工质量的切分信号**，
没有理由不用。

### 第二级：超长小节按句切分带重叠

```python
split_long_text(text, chunk_size=320, overlap=64) -> list[str]
```

- 按中文句末标点切句：`(?<=[。！？；\n])`；
- 逐句累积，超过 `chunk_size` 就成块；
- 新块开头保留上一块末尾 `overlap=64` 个字符。

**重叠是为了防止答案正好落在切点上**。比如"签收后 7 天内可申请退货。|节假日顺延。"
如果在竖线处切开，问"节假日怎么算"召回的块里只有"节假日顺延"四个字，
缺少"签收后 7 天内"这个上下文，模型答不清楚。64 字的重叠让后一块带上前文。

还有一个兜底：单句本身就超长时（少见，比如一整段没有标点的参数表格），
按 `chunk_size` 强行切，防止死循环。

### 标题拼进正文

```python
prefix = f"{meta.title} - {section_title}\n"
pieces.append(prefix + part)
```

每个块的内容是 `"商品名/文档名 - 小节标题\n正文"`。

为什么要拼：

- **BM25 受益**：小节标题里的词（"退货运费""保修期限"）是高区分度的检索词，
  不拼进去 BM25 就搜不到它们。
- **向量受益**：标题词汇参与哈希/嵌入，提升主题相关性。
- **模型受益**：上下文里带标题，模型知道这段话属于哪个主题，不容易跨主题串联。

代价是重复的标题文本会略微拉低 BM25 的 IDF 区分度（同一文档的所有块都含文档标题）。
在 6 篇文档的规模上这个影响可以忽略；语料大了可以考虑只拼小节标题。

## 3. chunk_id 的设计

```python
chunk_id = f"{meta.doc_id}#{seq}"  # 例如 aftersale_return_policy#2
```

三个性质，缺一不可：

1. **稳定可重建**：同样的文档内容 + 同样的 `chunk_size` → 同样的 chunk_id。
   这是索引幂等的前提（`upsert` 靠 id 去重）。
2. **人类可读**：出现在引用里是给人看的。`aftersale_return_policy#2` 一眼就知道
   来自哪篇文档的第几块；用 UUID 就得反查。
3. **可反查**：引用校验要用 chunk_id 在本次检索结果里查 `Chunk` 对象，
   取出 doc_id、title、snippet 回填给前端。

**注意它随 `chunk_size` 变化**。改了切块参数，所有 chunk_id 都会变，
历史对话里的引用就指不回去了。这也是评测集按**文档粒度**（doc_id）标注而不是块粒度的原因——
见 [EVALUATION](EVALUATION.md)。

## 4. 参数怎么选

| 参数 | 默认 | 影响 |
|------|------|------|
| `SMA_CHUNK_SIZE` | 320 | 太小→上下文碎片化，模型拼不出完整答案；太大→一个块里混多个主题，噪声多，Precision 降 |
| `SMA_CHUNK_OVERLAP` | 64 | 太小→答案在切点上被劈开；太大→索引膨胀、重复内容挤占 top_k 名额 |

320 这个值是针对中文的：中文一个字约等于英文 1.5–2 个 token，320 字大致对应
500–600 token，是一个"能完整承载一个小节"又不至于混入其他主题的量级。

> **没有做参数网格搜索。** 6 篇文档的语料上，K=4 时四种检索模式的 Recall 全是 1.0
> （天花板效应，见 [EVALUATION](EVALUATION.md)），切块参数的变化在这个语料上
> 测不出有意义的差异。要真正调这组参数，需要先把语料扩到几十到上百篇、
> 评测集扩到几百条，才有区分度。文档里不会给一个"实测最优 320"的假结论。

## 5. 向量化

### 5.1 两种实现

```python
class Embedder(ABC):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

| 实现 | 何时用 | 本质 |
|------|--------|------|
| `HashEmbedding`（默认） | 离线开发、测试、CI | 哈希词袋投影，**词面相似** |
| `ApiEmbedding` | 配了真实密钥 | OpenAI 兼容 `/embeddings`，真语义向量 |

### 5.2 HashEmbedding 是什么，不是什么

算法（`rag/embedding.py`）：

```python
tokens = _TOKEN_RE.findall(text.lower())  # 中文按单字，英文按词
for token in tokens:
    digest = sha256(token).digest()
    idx = int.from_bytes(digest[:4], "big") % dim  # 映射到固定下标
    sign = +1 if digest[4] % 2 == 0 else -1  # 随机符号，减少碰撞偏置
    vector[idx] += sign
return l2_normalize(vector)
```

这是经典的 **hashing trick**（feature hashing）：把词袋投影到 256 维稠密向量，
用随机符号抵消哈希碰撞带来的系统性偏置，最后 L2 归一化使余弦相似度等于点积。

**它不是语义向量，必须说清楚。** 它度量的是词面重叠：
"退货要钱吗"和"退货运费谁承担"共享"退货"一个字，相似度不会高；
"价格保护"和"退差价"没有任何共享词，相似度接近 0。真语义模型能认出后者是同义。

那它存在的意义是什么：**让向量检索链路在零外部依赖的情况下完整可测**。
索引、相似度计算、RRF 融合、评测脚本、CI——整条链路都能跑通、能回归。
如果默认走真实 API，那么没有密钥就跑不了任何测试，CI 也废了。

同义改写的检索能力，在默认配置下主要由 **BM25 的 jieba 分词 + 混合检索**承担
（jieba 会把"退差价"切成"退"/"差价"，和文档里的"差价"对上）。这也是为什么
默认 `SMA_RETRIEVAL_MODE=hybrid` 而不是 `vector`。

### 5.3 切换到真实 embedding

```bash
SMA_EMBEDDING_PROVIDER=deepseek_compatible \
SMA_EMBEDDING_API_KEY=sk-xxx \
SMA_EMBEDDING_BASE_URL=https://api.deepseek.com/v1 \
SMA_EMBEDDING_MODEL=text-embedding-ada-002 \
SMA_EMBEDDING_DIM=1536 \
uv run sma index --full          # 注意是 --full
```

**必须 `--full`**。原因：Chroma collection 的向量维度在创建时固定，
而且不同模型的向量空间完全不可比。把 1536 维的新向量和 256 维的旧向量放在
同一个 collection 里，要么直接报维度错误，要么（如果维度恰好相同）
算出完全无意义的相似度。增量模式只比对内容哈希，**不会**感知到 embedding 模型变了——
这是一个很容易踩的坑，所以写进了 README 和这里两处。

`ApiEmbedding` 的工程细节：

- **分批**：`embedding_batch_size=32`，避免单请求过大被拒；
- **按 index 重排**：`sorted(data, key=lambda d: d.get("index", 0))`——
  OpenAI 兼容接口不保证返回顺序与输入顺序一致，不排序会导致**向量和文本错配**，
  而且这种错误完全静默，索引看起来构建成功了，检索结果却是乱的；
- **数量校验**：返回条数 ≠ 输入条数直接抛错，不做"少了就补零"这种掩盖；
- **错误分类**：429 / 5xx → `LLMRateLimitError`（可重试），4xx → `EmbeddingError`（不可重试）。

### 5.4 维度

| provider | 默认 dim | 说明 |
|----------|---------|------|
| `hash` | 256 | 下限 32（低于此碰撞率太高，直接抛 `EmbeddingError`） |
| `deepseek_compatible` | 需手动设为模型实际维度 | 设错会在首次 upsert 时被 Chroma 拒绝 |

Chroma collection 的距离度量固定为 `cosine`（`metadata={"hnsw:space": "cosine"}`）。
`query()` 返回的相似度是 `1.0 - distance`，越大越相关——
这个转换在 `vector_store.py` 里做，上层拿到的永远是"越大越好"的分数，
不用关心底层是距离还是相似度。
