<!-- 文件作用：本项目完整业务流程参考文档，如实描述系统真实执行链路、状态流转、任务分发与结果汇总。仅作说明，不改变任何现有业务逻辑。深读业务流程时按需加载本文件。 -->

# Agent03 完整业务流程文档

本文档基于现有代码（agent/nodes.py、agent/graph_builder.py、agent/state.py、agent/tools.py、rag/）如实描述系统真实业务流程。

---

## 1. 系统总体架构

本项目为「LangGraph 显式状态机 + 混合检索 RAG + FastAPI 服务 + Docker 部署 + LangSmith 可观测」的跨境电商智能运营 Agent。

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI 接口层 (main.py)                                    │
│  · 输入校验、会话重置、超时控制、结果过滤、异常封装              │
│  · lifespan 一次性初始化：向量库 + BM25 + reranker + Agent图    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph 编排层 (graph_builder.py)                         │
│  · 节点注册 / 有向边 / 条件路由 / MemorySaver checkpoint      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  业务节点层 (nodes.py)                                       │
│  · 边界判断 / 知识判断 / 任务拆解 / 任务分发 / 结果汇总         │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────────┐      ┌──────────────────────────────┐
│  工具层 (tools.py)       │      │  RAG 检索层 (rag/)           │
│  爬虫/竞品/Listing/销售/客服│      │  Chroma + BM25 + RRF + Rerank│
└─────────────────────────┘      └──────────────────────────────┘
```

---

## 2. 完整执行链路（节点级）

启动入口：`pre_rag_retrieve` → 结束节点：`result_collect → END`。

```
        ┌────────────────────────────────────────────────────────────────────┐
        │                        LangGraph 执行链路                          │
        └────────────────────────────────────────────────────────────────────┘

 [pre_rag_retrieve]
    初始化/继承 AgentState，更新 user_query 与 trace_id
        │
        ▼
 [query_rewrite]  ──►  LLM 改写查询 → rewritten_query（补全模糊信息，不改变诉求）
        │
        ▼
 [boundary_judge]
    ├─ 硬关键词拦截（预测销量/下月销量/预估销量） → hit_boundary=True
    ├─ 同会话追问（历史 task_list 非空且 original_input≠query） → 放行
    └─ 否则 LLM 判断知识库是否有兜底声明
        │
        ▼  route_after_boundary
   ┌────┴─────┐
   │ hit=True │  hit=False
   ▼          ▼
 [result_collect]   [knowledge_judge]
                        │  need_retrieve=True / False
                        ▼  route_knowledge_judge
                 ┌──────┴───────┐
                 ▼              ▼
            [do_retrieve]   [task_parser]
            混合检索(8条)       │
                 │             │
                 └─────► [task_parser] ◄────┘
                             │
                             ▼  task_router
                      ┌──────┴────────┐
                      │ 有待执行/重试   │ 全部完成
                      ▼               ▼
                  [exec_task]    [result_collect]
                      │               │
                      └──► task_router │
                                       ▼
                                      END
```

---

## 3. 状态定义与流转规则

### 3.1 双层状态结构
- **顶层 `GraphState(MessagesState)`**：`messages`（对话消息，原生 reducer 追加）+ `agent_state`（业务对象）。
- **业务对象 `AgentState`**：全部业务字段的强类型容器。

### 3.2 关键流转规则（红线，改动时不可违反）
| 规则 | 说明 |
| ---- | ---- |
| 嵌套对象不可原地修改 | `agent_state.xxx = ...` 禁止，必须 `model_copy(update=...)` 或 `update_agent_state()` |
| 路由函数只返回字符串 | task_router / route_after_boundary / route_knowledge_judge 返回路由目标，不返回状态 |
| 节点返回状态更新字典 | 普通节点返回 `{"agent_state": ...}`，必要时附 `messages` |
| 消息由顶层维护 | messages 统一在 GraphState 顶层追加，AgentState 不保存消息 |

### 3.3 各节点对状态的读写

| 节点 | 读 | 写 |
| ---- | ---- | ---- |
| pre_rag_retrieve | messages[-1] | user_query、trace_id |
| query_rewrite | user_query | rewritten_query |
| boundary_judge | user_query、rewritten_query、retrieved_chunks、task_list、original_input | hit_boundary、boundary_answer |
| knowledge_judge | rewritten_query | need_retrieve、answer_source |
| do_retrieve | rewritten_query | retrieved_chunks |
| task_parser | user_query | task_list、current_task_type、current_task_index、error_msg、retry_count |
| exec_task | task_list、current_task_index、current_task_type、retry_count | 各类结果 / error_msg、retry_count+1 / 下标推进 |
| result_collect | 各任务结果、task_type、hit_boundary | final_answer、messages（追加 AIMessage） |

---

## 4. 边界判断（boundary_judge_node）

判断优先级（从上到下）：
1. **硬关键词拦截**：命中「预测销量 / 下月销量 / 下个月销量 / 预估销量」→ `hit_boundary=True`，返回固定兜底话术。
2. **同会话追问放行**：历史 `task_list` 非空 且 `original_input` 非空 且 不等于当前 query → 放行进入完整编排。
3. **LLM 兜底判断**：基于召回文本判断知识库是否含明确「无法回答/超出能力」声明；只有原文明确兜底才拦截，否则放行。重试最多 2 次，失败默认放行。

---

## 5. 知识检索判断（knowledge_judge_node）

LLM 判断 `need_retrieve`：
- **需要检索（True）**：跨境电商商品/参数/规格/材质、销售数据、FAQ、售后订单（退款/退货/物流/清关/质保等）→ 走 `do_retrieve`，`answer_source=platform_knowledge`。
- **不需要检索（False）**：天气、就业、机票等完全无关问题 → 直接 `task_parser`，`answer_source=llm_knowledge`。
- 无法确定时保守选 True。LLM 调用异常默认 True（防止售后类问题漏检索）。

---

## 6. 任务拆解（task_parser_node）

LLM 根据当前用户问题输出 JSON：`{"task_list": [...], "current_task_type": "..."}`。解析失败兜底为单条客服任务。

### 6.1 五大任务类型判定规则

| 任务类型 | 触发条件 | 典型任务列表 |
| ---- | ---- | ---- |
| customer_service | 商品 FAQ、参数说明、商品概况、爆款情况、笼统销售表现、纯闲聊/记忆类 | `["知识库检索并生成客服答案"]` / `["对话问答，基于对话历史生成回答"]` |
| analysis | 竞品对比、市场分析 | 拆分成多步子任务（如：收集竞品基础信息、分析优劣势、市场机会） |
| listing | 需要商品标题、五点描述 Listing 文案 | 生成 Listing |
| sales | **同时出现 SKU/商品编号 AND 具体年月（如 2026-08）** | 查询销售数据 |
| product_fetch | 传入商品 URL 要求抓取 | `["抓取商品网页信息"]` |

### 6.2 任务路由（task_router）
```
重试次数 >= MAX_TASK_RETRY(2)  → end
有 error_msg 且 未超限 且有任务  → exec_task（重试当前）
current_task_index < len(task_list) → exec_task（继续）
否则 → end（全部完成）
```

---

## 7. 任务执行分发（exec_task_dispatcher）

按 `current_task_type` 分发到工具层，统一异常捕获，失败写 `error_msg` + `retry_count+1`。

### 7.1 竞品分析（analysis）
1. LLM 将用户查询翻译为英文搜索关键词。
2. 构造 `books.toscrape` 搜索 URL；品类不在图书站点则返回空 + 提示。
3. `fetch_product_list` 抓取最多 3 个商品（防站点限流）。
4. 抓取失败 → 回退单次竞品分析；成功 → 并行执行全部子任务（共享同一份数据，避免重复爬虫）。
5. 一次完成全部子任务：下标直接跳到 `len(task_list)`。

### 7.2 商品抓取（product_fetch）
- 商品缓存复用：若 `product_fetch_completed=True` 且有 `product_raw_info`，跳过爬虫直接推进下标。
- 否则自动从 rewritten_query / user_query 提取 http(s) 链接 → `product_info_fetch` 抓取 → `save_product_to_knowledge` 保存 txt 到知识库 → 写入 `product_raw_info` 并标记完成。

### 7.3 Listing 生成（listing）
- 优先复用已抓取的 `product_raw_info`，无则抛异常（禁止用 user_query 直接爬虫）。
- `listing_generate` 生成英文标题/五点/广告文案，Prompt + Pydantic + 业务规则三重校验，失败重试最多 2 次。

### 7.4 客服问答（customer_service）
- 纯对话记忆任务：读取完整对话历史，LLM 直接自然回答，不检索知识库。
- 普通客服问答：`customer_service_qa(question, retrieved_docs)` 基于 RAG 召回结果 + LLM 生成，禁止编造知识库外信息。

### 7.5 销售查询（sales）
- `sales_data_query` 自然语言提取 product_id + month → 读取 `rag/sales_data.csv`（utf-8-sig，兼容 gbk）→ 精确匹配返回销量/营收/转化率。
- 无法解析出 SKU+月份或记录不存在 → 抛异常走重试/兜底。

---

## 8. 结果汇总（result_collect_node）

1. 设置来源标签：`【基于电商平台知识库回答】` / `【大模型原生知识回答】`。
2. 按 task_type 只选取当前任务原始素材（不拼接全部内容）：
   - hit_boundary → boundary_answer
   - product_fetch → product_raw_info
   - listing → listing_content
   - analysis → competitor_report
   - sales → sales_data
   - customer_service → customer_qa_result
3. 素材为空/空字典 → 直接返回兜底「暂时没有找到相关信息。」，不送 LLM。
4. 非空 → LLM 严格基于素材润色排版（禁止引入外部信息、删除内部标记），失败兜底直接输出原始素材。
5. 追加 AIMessage 到 messages，写 final_answer。

---

## 9. 服务请求处理流程（main.py /api/agent/run）

```
收到 POST /api/agent/run {user_query, session_id?}
   → 校验非空、长度 ≤800
   → 生成 trace_id
   → 读取 checkpoint 历史消息（get_state）
   → 拼接历史 + 本次 user 消息
   → 重置 agent_state（清历史任务数据，保留历史消息实现多轮记忆）
   → agent_graph.ainvoke(payload, config)（60s 超时）
   → 提取 final_answer（优先 agent_state.final_answer → competitor_report → 最后一条 AIMessage）
   → 正则清洗子任务标记
   → 过滤内部字段（trace_id/retry_count/task_list 等）
   → 返回 {code, msg, trace_id, cost_seconds, final_answer, data}
```
