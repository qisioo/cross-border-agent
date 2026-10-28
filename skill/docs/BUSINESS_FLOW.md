<!-- 文件作用：本项目完整业务流程参考文档，如实描述系统真实执行链路、状态流转、复合意图切分与反思、结果汇总与输出反思。仅作说明，不改变任何现有业务逻辑。深读业务流程时按需加载本文件。 -->

# Agent03 完整业务流程文档

本文档基于现有代码（agent/nodes.py、agent/graph_builder.py、agent/state.py、agent/tools.py、agent/mcp_tools.py、rag/）如实描述系统真实业务流程。

---

## 1. 系统总体架构

本项目为「LangGraph 显式状态机统一链路 + 混合检索 RAG + MCP 工具 + FastAPI 服务 + Docker 部署 + LangSmith 可观测」的跨境电商智能运营 Agent。

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI 接口层 (main.py)                                    │
│  · 输入校验、会话重置、超时控制、结果过滤、异常封装              │
│  · lifespan 初始化：Skill规则预加载 + 租户注册表 + Agent图      │
│  · 多租户/对话级知识库管理接口 + /api/agent/run + SSE 流式      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph 编排层 (graph_builder.py)                         │
│  · 6 节点统一链路 / 条件路由 / MemorySaver checkpoint         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  业务节点层 (nodes.py)                                       │
│  · 意图切分 / 反思校验 / 逐子意图执行 / 结果汇总 / 输出反思     │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────────┐      ┌──────────────────────────────┐
│  工具层 (tools.py)       │      │  RAG 检索层 (rag/)           │
│  爬虫/竞品/Listing/销售/客服│      │  Milvus+BM25+RRF+Rerank     │
└─────────────────────────┘      └──────────────────────────────┘
                │
                ▼
┌─────────────────────────┐
│  MCP 工具层 (mcp_tools)  │
│  get_weather/get_time    │
│  (连独立 mcp_demo server)│
└─────────────────────────┘
```

---

## 2. 完整执行链路（节点级）

启动入口：`pre_rag_retrieve` → 结束节点：`output_reflection → END`。单意图与复合意图完全同构，统一走同一条链路。

```
        ┌────────────────────────────────────────────────────────────────────┐
        │                        LangGraph 统一链路                           │
        └────────────────────────────────────────────────────────────────────┘

 [pre_rag_retrieve]
    初始化/继承 AgentState，更新 user_query 与 trace_id
        │
        ▼
 [intent_split]
    LLM 复合意图切分：原子粒度 + 分组标记 + 结合原句补全
    再按两层判断类型（knowledge 标 business / non_knowledge）
        │
        ▼
 [reflection_check]
    反思校验：编造 / 曲解换对象 / 分组是否合理 → 修正
    业务词规则兜底：漏判的业务词子意图强制改回 knowledge
        │
        ▼  route_after_split（统一路由）
 [intent_execute]
    逐子意图独立执行：
       knowledge → 按 business 路由到 销量/竞品/listing/抓取/客服
       non_knowledge → detect_tool_need 命中天气/时间走 MCP，
                        否则走纯对话（LLM+对话历史）
        │
        ▼
 [result_collect]
    按 group_id 分组 → 同组子结果联合衔接推理 → 大模型整理成连贯回答
        │
        ▼
 [output_reflection]
    对照原句+子意图+子结果检查回答是否合格
        │
        ▼  route_after_output_reflection
   ┌────┴─────────┐
   │ 未通过且≤1次  │  通过 / 达上限
   ▼              ▼
 [result_collect] END（达上限附加"回答可能不完整"提示）
```

---

## 3. 状态定义与流转规则

### 3.1 双层状态结构
- **顶层 `GraphState(MessagesState)`**：`messages`（对话消息，原生 reducer 追加）+ `agent_state`（业务对象）。
- **业务对象 `AgentState`**：全部业务字段的强类型容器，含复合意图与输出反思字段。

### 3.2 关键流转规则（红线，改动时不可违反）
| 规则 | 说明 |
| ---- | ---- |
| 嵌套对象不可原地修改 | `agent_state.xxx = ...` 禁止，必须 `model_copy(update=...)` 或 `update_agent_state()` |
| 路由函数只返回字符串 | route_after_split / route_after_output_reflection 返回路由目标，不返回状态 |
| 节点返回状态更新字典 | 普通节点返回 `{"agent_state": ...}`，必要时附 `messages` |
| 消息由顶层维护 | messages 统一在 GraphState 顶层追加，AgentState 不保存消息 |
| 意图独立执行 | 每个子意图单独路由执行，结果进 sub_results，最后统一汇总 |
| 输出反思有上限 | output_reflection_retry 最多 1 次，防止死循环 |

### 3.3 各节点对状态的读写

| 节点 | 读 | 写 |
| ---- | ---- | ---- |
| pre_rag_retrieve | messages[-1]、agent_state | user_query、trace_id（多轮继承历史状态） |
| intent_split | user_query | intent_list、sub_results=[], sub_executed=False |
| reflection_check | user_query、intent_list | intent_list（可能修正）、sub_results、sub_executed |
| intent_execute | intent_list、sub_results、user_query、messages | sub_results、sub_executed |
| result_collect | sub_results、user_query、output_reflection_issues、messages | final_answer、messages（追加/替换 AIMessage） |
| output_reflection | user_query、final_answer、intent_list、sub_results、messages | output_reflection_issues、output_reflection_retry、final_answer（达上限时加提示） |

---

## 4. 复合意图切分（intent_split_node）

### 4.1 切分规则（LLM）
把用户整句拆成多个「原子级子意图」，每个子意图单一诉求、单一来源，能从原话直接对应：
1. **切分到原子粒度**：只要原话隐含多个可独立获取的信息诉求就拆分。如"天气会不会影响物流、大概几天能到"隐含"查天气"+"查物流规则"两个诉求 → 拆成两个子意图，但同属一个上层问题，group 编号相同。
2. **组的概念**：需联合推理的子意图（如"天气"+"物流规则"一起回答"天气是否影响物流"）用相同 group_id；相互独立的用不同 group_id。
3. **补全省略**：子意图存在省略时，必须结合用户原句理解指代关系，补全为主语/宾语使其能独立理解。补全只能用原句已出现的对象，禁止凭空编造/换对象（如原句"蓝牙耳机"绝不能补成"牙刷/电脑"）。
4. **不拆碎片**：不要把"大概多少天""多久能到"这类时间碎片拆成独立子意图，必须并入所属主诉求。
5. 整句单一意图 → 只输出 1 个子意图。

示例："武汉今天天气怎么样，给我推荐一个蓝牙耳机，如果我下单的话武汉的天气会不会影响物流运输，大概多少天能到"
→ 拆分为多个子意图，其中"天气"与"物流几天能到"同组，供后续联合推理。

### 4.2 类型判断（两层，不用 LLM 判类型）
- **knowledge**：命中业务词（商品/物流/售后/销量等），需要读知识库；额外标注 business=sales/analysis/listing/product_fetch/customer_service。
- **non_knowledge**：其余全部（天气/时间/闲聊/通用知识），不读知识库；是否调 MCP 工具由 exec_intent_node 内 detect_tool_need 决定。
- 时间/日期类（几点/日期/星期几）优先走 non_knowledge，避免被"几天|多久"等业务词误命中。

### 4.3 兜底
拆分失败或完全拆不出子意图 → 把整句当作 1 个子意图，保证链路不中断。

---

## 5. 反思校验（reflection_check_node）

### 5.1 LLM 反思
让大模型审视切分结果是否正确，检查三点：
1. **是否编造**：每个子意图都能在原句找到对应内容？
2. **是否曲解/换对象**：补全部分有没有把主语/宾语换成原句没有的对象？
3. **分组是否合理**：需联合推理的是否同组、相互独立的是否不同组？
有问题 → 返回修正后的完整切分结果；没问题 → 原样保留。

### 5.2 业务词规则兜底（新增）
LLM 反思后，再用业务词列表（退换货/物流/售后/优惠/价格/仓库/预售/大件/商品/订单等）强制校验一次：
- 若子意图含业务词但 type 不是 knowledge → 强制改成 knowledge，并补 business（默认 customer_service）。
- 目的：LLM 在混合问题中容易漏判业务词，规则兜底更稳。

### 5.3 降级
反思失败 → 保留原切分结果，不阻断链路，仅记录日志。

---

## 6. 逐子意图执行（exec_intent_node）

对 intent_list 每个子意图独立执行，结果收集进 sub_results。任一子意图失败仅记录错误，不中断整条链路。

### 6.1 knowledge 业务路由（内化旧 task_parser 能力）

| business | 执行逻辑 |
| ---- | ---- |
| sales | `sales_data_query(user_query=子意图)`，自然语言提取 SKU+月份 → 读 CSV 返回 |
| analysis | LLM 提取英文关键词 → 构造 books.toscrape 搜索 URL → `fetch_product_list(limit=3)` 抓取共享数据 → 并行 `competitor_analysis_parallel` 分析 |
| listing | 优先复用 `product_raw_info`（无则回退客服问答，提示需商品资料）→ `listing_generate` 三重校验 |
| product_fetch | 从子意图/原句提取 URL → `product_info_fetch` 抓取 → `save_product_to_knowledge` 保存到知识库 |
| customer_service（默认） | 从注册表取本对话检索器 → `hybrid_search(子意图, top_k=4)` → `customer_service_qa` |

### 6.2 non_knowledge 路由
1. `detect_tool_need(intent)` 判断是否命中天气/时间：
   - weather → `get_weather(city)`（MCP 取实时数据）
   - time → `get_time()`（MCP 取实时数据）
2. 否则 → 纯对话：`ChatDeepSeek` + 对话历史，DeepSeek 原生知识直接回答（如"计算机就业"、闲聊）。

---

## 7. 结果汇总（result_collect_node）

统一出口，隐藏所有内部子任务/调试标记：
1. 按 `group_id` 分组（保持原始出现顺序）。
2. 每组内部组织成块，组间按顺序拼接。
3. 同组多个子意图说明是同一上层问题的不同来源，需联合做衔接推理（如"天气"+"物流规则"一起回答"天气是否影响物流"）。
4. 交给大模型整合成连贯、完整、面向用户的回答，覆盖所有分组、不暴露内部标记。
5. 若输出反思发现问题（`output_reflection_issues` 非空），重新整合时把问题反馈给大模型重点修正。
6. LLM 异常兜底：把子结果按组简单拼接，防止链路崩溃。
7. 追加/替换 AIMessage 到 messages，写 final_answer。

---

## 8. 输出反思（output_reflection_node）

校验最终回答是否真正覆盖用户原始问题：
1. 构造子意图清单 + 各子意图实际执行结果（真实查到的信息，不算编造）+ 最近几轮对话历史。
2. 让大模型对照检查：用户问的每个子意图回答里是否都体现、是否答非所问、信息来自历史/子结果不算编造、子意图结果"没有找到"要诚实说明。
3. 通过 → 清空 issues，路由结束。
4. 不通过且未超限（retry<=1）→ 记录 issues，路由回 result_collect 重新整合。
5. 不通过且已达上限 → 在回答前附加"⚠️ 以上回答可能存在遗漏或考虑不完整"提示，然后结束（防死循环）。

---

## 9. 服务请求处理流程（main.py）

### 9.1 POST /api/agent/run
```
收到 POST /api/agent/run {user_query, tenant_id, conversation_id?, session_id?}
   → 校验非空、长度 ≤800
   → 生成 trace_id
   → 构造 thread_id = {tenant_id}:{conversation_id}:{session_id}（对话级隔离）
   → 前置重置 checkpoint 里的 agent_state（清历史任务数据，保留历史消息）
   → agent_graph.ainvoke(payload, config)（60s 超时）
   → 提取 final_answer（优先 final_answer → competitor_report → 最后一条 AIMessage）
   → 正则清洗子任务标记
   → 过滤内部字段
   → 返回 {code, msg, trace_id, cost_seconds, final_answer, data}
```

### 9.2 POST /api/agent/stream（SSE）
用 LangGraph 原生流式能力（`stream_mode="updates"`）推送节点执行过程（start → node → delta → done），回答按 token 级切分模拟打字机效果推送。

### 9.3 多租户/对话级知识库接口
- 企业级：上传、构建进度、企业知识库。
- 对话级：对话 CRUD、对话知识库上传/列出/删除/重建，知识库按 `tenant_id` 与 `conversation_id` 双层隔离。

---

## 10. 自测方式

```
uv run python -m agent.nodes          # 节点与路由自测
uv run python -m agent.graph_builder  # 构图与统一链路自测
uv run python -m agent.mcp_tools      # MCP 天气/时间识别与调用自测
```
