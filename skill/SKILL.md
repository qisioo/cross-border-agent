---
name: agent03
description: 跨境电商智能运营 Agent（Agent03）项目的开发与维护 Skill。用于对本项目进行代码维护、功能扩展、Bug 定位、业务理解、运行部署、接口调用。项目基于 LangGraph 显式状态机统一链路编排（pre_rag_retrieve → intent_split → reflection_check → intent_execute → result_collect → output_reflection），覆盖竞品分析、商品信息抓取、Listing 文案生成、销售数据自然语言查询、知识库客服问答五大业务场景，支持复合意图切分+反思校验+输出反思，并具备 BM25+Milvus 混合检索 RAG（企业+对话级双层隔离）、Qwen-VL 多模态文档识别、MCP 天气/时间工具、FastAPI 服务封装（含 SSE 流式）、Streamlit 前端工作台、Docker 容器化等工程能力。触发词：Agent03、跨境电商运营助手、跨境智能运营、销售数据查询、竞品分析、Listing、客服问答、混合检索、多意图、复合意图。
---

# 跨境电商智能运营 Agent（Agent03）维护开发 Skill

<!-- 文件作用：本文件是 skill 包唯一主控入口（SKILL.md），存放项目核心信息与硬约束，并索引 docs/ 下各参考文档。Agent 匹配到本 skill 后优先读取本文件；遇到具体深度任务时再按需读取对应 docs 文档（渐进式加载）。不修改任何现有业务代码。 -->

---

## 一、项目定位与能力边界

本系统是垂直领域智能运营 Agent，面向跨境电商运营人员，将四类重复性工作自动化，并在统一链路中支持「闲聊 + 业务」复合意图的混合处理：

| 业务场景 | business 类型 | 说明 |
| ---- | ---- | ---- |
| 竞品分析 | analysis | 基于公开图书站点 books.toscrape 抓取竞品，LLM 生成分析报告 |
| 商品信息抓取 | product_fetch | 传入商品 URL，抓取标题、价格、卖点并保存到知识库 |
| Listing 文案生成 | listing | 基于商品信息生成英文标题、五点描述、广告文案（三重校验） |
| 销售数据查询 | sales | 自然语言提取 SKU+月份，读取本地 CSV 返回结构化结果 |
| 知识库客服问答 | customer_service | 基于 RAG 召回结果 + LLM 生成客服回复（默认业务） |
| 天气 / 时间 | non_knowledge | 日常对话命中天气/时间时，通过 MCP 工具取实时数据 |
| 闲聊 / 通用知识 | non_knowledge | 与业务无关的日常问题，DeepSeek 原生知识直接回答 |

> **硬性约束（用户明确要求，开发时必须遵守）：**
> 1. 现阶段的业务逻辑和功能能力**不需要删减和优化**，禁止自主删减、改写或重构既有功能。
> 2. 新增能力/文档时不得修改现有代码文件的既有逻辑与格式。
> 3. 所有修改必须**基于现有代码**进行，返回完整修改后的代码，不提供片段。
> 4. 本项目为本地运行项目，依赖由 uv / Docker 管理，不使用手动虚拟环境。
> 5. 写文件/改文件必须先经用户允许，禁止擅自创建、删除、覆盖项目文件。

---

## 二、技术栈与关键版本

- 语言/框架：Python ≥3.11、FastAPI、Pydantic v2
- Agent 编排：LangGraph（StateGraph 显式状态机 + MemorySaver checkpoint，thread_id 实现多租户+对话级隔离）
- LLM：LangChain + DeepSeek（`langchain-deepseek`，model=`deepseek-chat`）
- 检索：Milvus 向量库（多租户+对话级 collection `agent03_<tenant>[_<conv>]`）+ BM25（rank-bm25，k=5）+ RRF（rrf_k=60）+ BGE-reranker 重排序（阈值 0.15，动态 top-n 断层 0.15）
- Embedding：BAAI/bge-m3（CPU，HF 国内镜像 hf-mirror.com）
- 多模态：Qwen-VL（`qwen3.8-omni-flash`，dashscope）识别 PDF 内图片
- MCP：`langchain-mcp-adapters` 连接独立 MCP Server（`mcp_demo/server.py`，提供 get_weather/get_time）
- 服务：FastAPI，含 `text/event-stream` SSE 流式接口
- 部署：Docker（python:3.11-slim，uv 管理依赖）、uv
- 可观测：LangSmith（环境变量开关控制，非强依赖）

依赖清单以根目录 `pyproject.toml` 为准，所有软件包操作必须基于其中列出的依赖进行。

---

## 三、项目目录结构

```
Agent03/
├── main.py                  # FastAPI 主入口：lifespan 初始化、/api/agent/run、/api/agent/stream、/health、多租户/对话知识库管理接口
├── pyproject.toml           # 项目依赖与构建配置（清华源）
├── Dockerfile               # 镜像构建（分层缓存）
├── uv.lock                  # 依赖锁定文件
├── .env                     # 环境变量（含 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY，已 gitignore）
├── .gitignore               # 忽略 .venv、__pycache__、构建产物
├── README.md                # 项目说明书
├── skill/
│   ├── SKILL.md             # 本主控文件
│   ├── rules/               # 注入给 LLM 的固定规则分片（skill_loader 加载）
│   │   ├── customer_service.md
│   │   ├── listing.md
│   │   ├── result_collect.md
│   │   └── sales.md
│   └── docs/                # 参考文档（渐进式加载）
│       ├── BUSINESS_FLOW.md # 完整业务流程与节点细则
│       ├── RAG_SYSTEM.md    # 混合检索体系详细说明
│       ├── API_REFERENCE.md # 接口规范与参数说明
│       └── DEPLOY_GUIDE.md  # 运行部署与 LangSmith 配置手册
├── agent/
│   ├── state.py             # AgentState / GraphState 状态定义（含 intent_list/sub_results/输出反思字段）
│   ├── graph_builder.py     # LangGraph 构图：6 节点统一链路注册、条件路由
│   ├── nodes.py             # 全部业务节点与路由函数（意图拆分/反思/执行/汇总/输出反思）
│   ├── tools.py             # 业务工具集（爬虫/竞品/Listing/销售/客服）
│   ├── output_parsers.py    # LLM 输出解析与 Pydantic 校验
│   └── mcp_tools.py         # MCP 工具封装（get_weather/get_time，连独立 mcp_demo server）
├── rag/
│   ├── build_vector_db.py   # 知识库加载（含 Qwen-VL 图片识别）、切分、MD5 去重、向量库+BM25 构建
│   ├── hybrid_retriever.py  # 混合检索：BM25+Milvus+RRF+Rerank（动态 top-n）
│   ├── knowledge/           # 知识库文档（txt/md/csv/pdf，按企业/对话分目录）
│   │   └── default/8c6fd7b3/  # 企业 default 下某对话的知识库（对话级隔离）
│   ├── sales_data.csv       # 销售数据
│   ├── tenant_registry.py   # 租户检索注册表（多租户+对话级隔离）
│   └── conversation_store.py# 对话元数据持久化（conversations.json）
└── utils/
    ├── helper.py            # trace_id、update_agent_state、文本/字典工具
    └── skill_loader.py      # skill/rules 规则分片加载器（渐进式加载）
```

---

## 四、核心架构与完整执行链路

### 4.1 状态定义（agent/state.py）
- `AgentState`：业务状态，所有业务字段的强类型容器。**关键约束：嵌套 Pydantic 对象禁止原地修改，必须用 `model_copy(update=...)` 整体替换返回。**
- `GraphState(MessagesState)`：LangGraph 顶层状态，`messages` 由原生 reducer 追加；`agent_state` 为嵌套业务对象，更新必须整体替换。
- 状态更新统一走 `utils/helper.update_agent_state()`，自动过滤非法字段并打印告警。
- 复合意图相关字段：`intent_list`（切分后的原子子意图，含 type/business/group_id）、`sub_results`（各子意图执行结果）、`sub_executed`（是否已执行）；输出反思字段：`output_reflection_retry`、`output_reflection_issues`。

### 4.2 LangGraph 统一链路（agent/graph_builder.py）
当前真实链路为 **6 节点统一链路**（单意图与复合意图完全同构，不区分两条路）：

```
pre_rag_retrieve ──► intent_split ──► reflection_check
                                              │
                              route_after_split│
                                              ▼
                                         intent_execute
                                              │
                                              ▼
                                         result_collect
                                              │
                                              ▼
                                         output_reflection
                                              │
                         route_after_output_reflection│
                  ┌───────────────────────────┴────────────┐
                  │ issues 且 retry<=1                     │ 通过 / 达上限
                  ▼                                        ▼
            result_collect（重新整合）                  END
```

节点说明（均来自真实代码）：
- `pre_rag_retrieve_node`：入口。首轮初始化 AgentState，多轮则继承历史状态，更新本次 `user_query` 与 `trace_id`。
- `intent_split_node`：复合意图拆分（方案B两层）。LLM 把整句切分到「原子粒度」，每个子意图单一诉求、单一来源，结合原句补全省略语义，为同属一个上层问题的子意图打相同 `group_id`；再按两层判断类型：命中业务词 → `knowledge`（额外标注 business=sales/analysis/listing/product_fetch/customer_service），其余（天气/时间/闲聊/通用知识）→ `non_knowledge`。拆分失败降级为整句单意图。
- `reflection_check_node`：反思校验。让大模型审视切分结果是否正确（是否编造、是否曲解/换对象、分组是否合理），有问题则修正；随后用**业务词规则**强制兜底（把漏判的业务词子意图改回 knowledge）。反思失败保留原切分，不阻断链路。
- `route_after_split`：条件路由，所有意图（单/复合）统一进入 `intent_execute`。
- `exec_intent_node`：逐子意图执行（内化旧 task_parser/exec_task 能力）。`knowledge` 按 business 路由到具体工具（sales/analysis/listing/product_fetch/customer_service）；`non_knowledge` 内部用 `detect_tool_need` 判断——命中天气/时间调 MCP 工具取实时数据，否则走纯对话（LLM+对话历史）。任一子意图失败仅记录错误，不中断链路。
- `result_collect_node`：统一出口。按 `group_id` 分组，同组子结果联合做衔接推理（如「天气+物流」一起回答「天气是否影响物流」），覆盖全部组后由大模型整理成连贯回答。
- `output_reflection_node`：输出反思校验。对照用户原句 + 子意图 + 子结果，检查回答是否覆盖了每个子意图、有没有答偏/编造；不通过则记录问题并路由回 `result_collect` 重新整合（最多 1 次），超限则在回答前附加「回答可能不完整」提示。
- `route_after_output_reflection`：条件路由。`issues` 非空且未超限 → 回 `result_collect`；通过或达上限 → `END`。

### 4.3 五大业务类型触发逻辑（exec_intent_node 内化判断）
1. **customer_service**（默认）：商品 FAQ、参数说明、商品概况、爆款情况、笼统销售表现、售后政策。
2. **analysis**：竞品对比、市场分析，一次抓取共享数据、并行做 LLM 分析。
3. **listing**：需要商品标题、五点描述 Listing 文案（优先复用已抓取的 `product_raw_info`）。
4. **sales**：**必须同时出现 SKU/商品编号 AND 具体年月（如 2026-08）**；仅商品名称、无 SKU 或无月份，归 customer_service。
5. **product_fetch**：传入商品 URL，要求抓取商品信息并保存到知识库。

---

## 五、关键开发红线（改动时不可违反）

1. **嵌套状态修改**：禁止 `agent_state.xxx = value`，必须 `model_copy(update=...)` 或 `update_agent_state(state, xxx=...)`。
2. **路由函数只返回字符串**：route_after_split、route_after_output_reflection 返回路由目标，不返回状态字典；普通节点才返回状态更新字典。
3. **复合意图切分**：必须结合用户原句补全省略语义，禁止把原句没有的对象（如把「蓝牙耳机」补成「牙刷/电脑」）；不要把「程度/时间碎片」（如「大概多少天」）拆成独立子意图。
4. **sales 判定**必须同时有 SKU 与年月，否则归 customer_service。
5. **竞品分析并发**：一次抓取共享商品数据，子任务仅并行做 LLM 分析，避免重复爬虫触发站点限流。
6. **listing 三重校验**：Prompt + PydanticOutputParser + `parse_listing_output` 业务规则校验（标题 60~120 字符全英文、五点固定 5 条、广告 ≤80 字符全英文），失败自动重试最多 2 次。
7. **输出反思重试**：`output_reflection_retry` 最多 1 次；达上限后在回答前附加「回答可能不完整」提示后结束，防止死循环。
8. **接口并发重置**：每次 `/api/agent/run` 与 `/api/agent/stream` 前置 `agent_graph.update_state(config, {"agent_state": clean_agent_state})` 清残留任务数据，但保留历史消息实现多轮记忆。
9. **LLM 输出解析**统一走 `safe_parse_json`（自动剥离 markdown 代码块），失败返回 None 并走兜底分支。
10. **BM25 索引仅内存态**：进程销毁即丢失，服务重启必须重新触发知识库构建/重建（上传或 rebuild 接口）；返回的 HybridRetriever 实例必须由 `TenantRetrieverRegistry` 全程持有。
11. **HF 镜像前置**：`HF_ENDPOINT=https://hf-mirror.com` 必须放在所有 huggingface 相关 import 之前（build_vector_db.py 顶部已处理）。
12. **Qwen-VL 密钥**：PDF 图片识别需 `DASHSCOPE_API_KEY`，从环境变量读取，不硬编码；未配置则跳过图片识别不报错。
13. **MCP 工具解耦**：`get_weather/get_time` 只通过 `agent/mcp_tools.py` 调用，工具实现都在外部 `mcp_demo/server.py`，Agent 内不承载工具实现。
14. **环境密钥统一从 .env 读取**：所有项目 API key（DEEPSEEK_API_KEY、DASHSCOPE_API_KEY 等）只从 `.env` 环境获取，代码不硬编码；`.env` 已 gitignore。

---

## 六、附属文档索引（渐进式加载）

遇到以下深度任务时，按需读取对应文档，无需全部加载：

| 任务类型 | 对应文档 | 内容 |
| ---- | ---- | ---- |
| 理解业务流程、节点细节、状态流转、意图拆分/反思/输出反思 | `docs/BUSINESS_FLOW.md` | 完整统一链路、六节点细节、复合意图切分+反思、各节点读写、服务请求处理 |
| RAG 检索、知识库构建、混合检索调优、多模态 | `docs/RAG_SYSTEM.md` | BM25+Milvus+RRF+Rerank 细节、动态 top-n、对话级隔离、Qwen-VL 图片识别、MD5 去重、内存索引特性 |
| 接口调用、参数格式、错误码 | `docs/API_REFERENCE.md` | 多租户/对话知识库管理接口、/api/agent/run、/api/agent/stream(SSE)、/health、错误码 |
| 本地运行、Docker 部署、LangSmith 配置 | `docs/DEPLOY_GUIDE.md` | 运行步骤、环境变量、Milvus/MCP 依赖、镜像构建、容器启动 |

---

## 七、快速排错指南

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| 接口报 402 Insufficient Balance | DeepSeek 账号余额不足 | 到 DeepSeek 开放平台充值，与代码无关 |
| 回答「知识库中没有找到相关信息」 | 后端重启后对话检索器未注册（BM25 内存索引丢失） | 调用该对话的 rebuild/上传接口触发知识库重建，注册检索器后再问 |
| 天气/时间查询无返回或报 MCP 不可用 | 外部 MCP Server（`mcp_demo/server.py`）未启动 / 环境不匹配 | 用 `mcp_demo` 的 .venv 解释器启动 server，确认进程在跑 |
| 查询「时间/日期」被当业务词误判 | 时间类子意图在切分时被业务词命中 | 已内置 `_TIME_PAT_EARLY` 优先走 non_knowledge，若仍误判检查意图拆分日志 |
| PDF 图片识别无内容 | 未配置 `DASHSCOPE_API_KEY` | 在 `.env` 配置 DASHSCOPE_API_KEY，重启服务 |
| 启动阶段长时间卡住 | 首次加载 BGE-M3/reranker 需联网下载 | 属正常现象，等待；可挂载本地模型目录加速 |
| Milvus 连接失败 | 本地 Milvus standalone（Docker）未启动 | 启动 Milvus 容器（端口 19530），再重启后端 |
| LangSmith 页面看不到 Trace | 缺少环境变量 / 项目名不匹配 | 确认 LANGSMITH_TRACING/ENDPOINT/API_KEY/PROJECT 四项，PROJECT 与平台一致 |
| 爬虫抓取失败/超时 | 站点限流 | 依赖既有重试、延时、并发控制逻辑，不擅改 |
| 外部无法访问接口 | 监听地址 / 端口映射不对 | FastAPI 监听 0.0.0.0，Docker 用 `-p 8000:8000` 映射 |

---

## 八、扩展指引（新增功能时遵循既有模式）

1. **新增业务场景**：在 `agent/tools.py` 实现工具函数并加入 `__all__` → 在 `agent/nodes.py` 的 `intent_split_node`（`_KNOWLEDGE_PAT` + `_re_match_business`）增加 business 类型判定 → 在 `exec_intent_node` 增加对应 business 分支 → 在 `agent/state.py` 增加结果字段。
2. **新增知识库文档**：通过前端/接口上传 txt/md/csv/pdf 到对应企业或对话目录，异步构建向量库+BM25 并注册。
3. **新增外部数据源**（如替换爬虫站点）：修改 `tools.py` 中的 URL 与页面选择器，保持返回 schema 不变。
4. **新增 MCP 工具**：在外部 `mcp_demo/server.py` 实现工具 → 在 `agent/mcp_tools.py` 增加调用函数与 `detect_tool_need` 识别。
5. **性能/稳定性**：遵循既有模式——重型资源全局单例、工具级异常捕获、重试熔断、降级兜底，不引入破坏性改动。

---

## 九、自测方式

各核心模块均内嵌 `if __name__ == "__main__"` 自测代码，可直接运行验证：
```
uv run python -m agent.state          # 状态模型自测
uv run python -m agent.graph_builder  # 构图与链路自测
uv run python -m agent.nodes          # 节点与路由自测
uv run python -m agent.tools          # 工具自测（默认 Mock 模式，爬虫块已注释）
uv run python -m agent.output_parsers # 解析校验自测
uv run python -m agent.mcp_tools      # MCP 工具识别与调用自测
uv run python -m rag.hybrid_retriever # 混合检索自测
uv run python -m rag.build_vector_db  # 知识库构建自测
uv run python -m utils.helper         # 工具函数自测
```
> 注意：`agent.tools` 与 `rag.build_vector_db` 的自测会触发真实爬虫/模型加载，执行较慢，按需开启注释块。
