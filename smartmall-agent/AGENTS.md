# AGENTS.md —— 后续开发必须遵守的工程规则

本文件给后续接手的开发者（人或 AI）看。违反这些规则的改动会破坏项目的核心不变量。

## 一、不可破坏的不变量

1. **业务结论只能来自代码，不能来自模型。** 退货资格、订单归属、库存是否充足——
   这类结论必须由 `tools/business_rules.py` 或工具的 `_run` 算出来。
   任何"让模型判断用户能不能退货"的改动都是 bug。
2. **身份只能来自认证层。** `ToolContext.user_id` 只能从 `AgentState["user_id"]` 取，
   而后者只能从 HTTP 头取。任何从聊天内容、模型输出 slots 里取身份的路径都是漏洞。
3. **模型不能指定工具名。** 模型输出意图，意图→工具的映射是 `INTENT_TOOL_MAP` 常量表。
   把工具名交给模型等于打穿能力最小化。
4. **引用必须真实。** 任何绕过 `validate_citations` 的回答生成路径都是 bug。
   伪造引用剔除后一条不剩 → 必须降级为 `no_answer`。
5. **伪实现不得伪装成成功。** 没对接的能力就抛错并说明
   （参照 `RefundTool._run` 的写法），不要返回假的成功结果。
6. **无密钥可测。** 任何新功能都必须有 mock 路径。
   `uv run pytest` 必须在无网络、无密钥时全绿。
7. **索引幂等。** 重复执行 `sma index` 写入量必须为 0。
   改切块或索引逻辑后，`test_index_idempotent` 必须仍然通过。
8. **失败往安全方向降级。** 模型挂了降级到"不触发工具的分支"，不是反过来。
   但索引为空（`KnowledgeBaseEmptyError`）**必须暴露**，不许兜底成"没找到答案"。
9. **图中不得加回边。** `agent/workflow.py` 的图必须保持无环，路径长度有上界。
10. **所有排序必须有 tie-break。** 分数相同时按 `chunk_id` 排，否则结果不可复现，
    评测就失去意义。

## 二、编码规范

- 遵循 PEP 8，行宽 120，由 `ruff format` 统一（不要手动调格式）。
- 公共接口必须有类型标注；关键输入输出用 Pydantic 模型。
- **禁止空 except**。兜底捕获必须：记日志 + 返回明确的失败状态 + 注释说明为什么要兜底。
  现有的两处 `# noqa: BLE001` 都写了理由，照这个标准来。
- **每个 `# noqa` 必须在同一处写明理由。** 没有理由的 noqa 等于把规则关掉。
- 所有外部调用必须有超时，并把异常映射为 `RetryableError` / `PermanentError` / `UserError`。
- 重试必须有次数上限和退避。
- 注释写"为什么"，不写"做了什么"。决策性说明放 docstring；架构级决策放 `TECH_DECISIONS.md`。
- 中文注释，与现有风格一致。
- 密钥类配置用 `SecretStr`，取明文走 `_reveal` 包装的 property。

## 三、改动流程

1. **先读相关文档**：改检索读 `RETRIEVAL.md`，改工具读 `TOOLS_AND_BUSINESS_RULES.md`，
   改图读 `AGENT_WORKFLOW.md`，碰权限读 `SECURITY.md`。
2. **先写测试再写实现**，尤其是失败路径和越权路径的测试。
3. **跑完整检查**：

   ```bash
   uv run ruff check . && uv run ruff format .
   uv run pytest --cov=sma
   uv run sma index && uv run sma eval
   ```

4. **评测不许回归**。行为用例当前 12/12，任何改动后必须仍然 12/12。
   **如果用例失败，先判断是代码错还是期望错——默认假设是代码错。**
   项目历史上两次行为用例失败（`b01` 无关问题被编答案、`b10` 注入未拦截）
   都是真实缺陷，修复方式是加 `agent/guards.py`，不是改期望值。
5. **同步文档**。改了行为就改文档。文档与代码不一致比没有文档更糟。
   带实测数字的文档（`RETRIEVAL.md`、`RERANKING.md`、`EVALUATION.md`）
   改了相关逻辑就要**重跑脚本更新数字**，不要手改。
6. **发现新缺陷写进 `ROADMAP.md` 的已知缺陷表**，不要留在脑子里。

## 四、添加新东西的位置

| 要加什么 | 加在哪 | 别忘了 |
|---------|-------|-------|
| 新知识文档 | `data/knowledge/*.md`（带 frontmatter） | `sma index`；考虑加评测用例 |
| 新业务工具 | `tools/business_tools.py` 继承 `BaseTool` → `build_registry` | `INTENT_TOOL_MAP`、`TOOL_REQUIRED_SLOTS`、评测用例；碰订单必须调 `assert_order_ownership` |
| 新意图 | `models.py:Intent` → `prompts.py` 意图清单 → `workflow.py:INTENT_ROUTES` | **必须加评测用例**（新意图容易抢走已有意图的分类） |
| 新 Embedding | `rag/embedding.py` 实现 `Embedder` → `llm/factory.py` 注册 | **换完必须 `sma index --full`** |
| 新向量库 | 替换 `ChromaVectorStore` | 接口：upsert / query / delete_doc / all_chunks / count / doc_versions / reset |
| 新重排器 | `rag/retriever.py` 实现 `Reranker` → 注入 `HybridRetriever` | 重跑 `scripts/eval_sweep.py` 并更新 `RERANKING.md` 的数据表 |
| 新 LLM 提供方 | `llm/` 实现 `LLMClient` → `llm/factory.py` 注册 | 用 `httpx.MockTransport` 测错误分类，**不要 mock 整个类** |
| 新配置项 | `config.py` → `.env.example` → README 配置表 | 密钥用 `SecretStr` |
| 新 API 端点 | `api/routes.py` + `api/schemas.py` | 管理类接口挂 `Depends(require_admin)`；错误用 `SMAError` 子类 |
| 新注入模式 | `agent/guards.py:INJECTION_PATTERNS` | 加对应评测用例；只放高置信度模式，避免误伤正常提问 |

## 五、特别容易踩的坑

1. **LangGraph 会静默丢弃未在 `AgentState` TypedDict 里声明的键。**
   新增跨节点传递的字段必须先声明，否则下游拿不到且不报错。
   这个坑真的踩过：引用校验永远拿到空列表，表现成"所有知识问答都被拒答"。
2. **换 embedding 模型后增量索引感知不到。** 必须 `--full`。
   维度相同的话错误是**静默**的——索引看起来成功了，相似度全是垃圾。
3. **改 `chunk_size` 会让所有 `chunk_id` 变化。** 历史对话里的引用会指不回去。
   这也是评测集按 doc_id 而不是 chunk_id 标注的原因。
4. **`min_score` 的量级随检索模式变化。** RRF 分数约 0.016 量级，
   向量相似度是 0–1 量级。把向量模式的阈值用在 hybrid 上会过滤掉全部候选。
5. **`relevance_guard` 的阈值不能凭感觉调。** 调低会让"无关问题被编答案"的缺陷回归，
   调高会误杀同义表达。改它必须跑评测集。
6. **工具的"资源不存在"话术必须和"无权访问"完全一致。**
   不一致就产生订单号枚举的侧信道。
7. **写操作必须自己重跑业务规则**，不能信任前一次预检查的结论
   （两次之间订单状态可能已变）。
8. **日志走 stderr。** 新增任何 print 或第三方库的 stdout 输出都会污染
   `sma eval > report.json`。jieba 就干过这事。
9. **时间相关的断言用 `mock_data.NOW`**，不用 `datetime.now()`，否则测试会在未来某天失败。
10. **`ruff format --check` 在 CI 里是 `|| true`（只提示），`ruff check` 是硬门禁。**
    不要把 lint 问题也当成"格式问题"糊过去。

## 六、不要做的事

- 不要为了让测试通过而修改评测期望值。
- 不要把 Mock / TODO / 未实现描述成已完成。
- 不要在文档里写没有实际跑出来的性能或质量数字。
- 不要引入 CrewAI 或其他多智能体框架（理由见 `TECH_SOLUTION.md` 1.4）。
- 不要给 Agent 增加 Shell、任意文件写入、任意 URL 访问的工具。
- 不要在应用层实现速率限制（网关的职责）。
- 不要未经授权重构既有结构；改动遵循最小 diff，保留原有注释与日志。
