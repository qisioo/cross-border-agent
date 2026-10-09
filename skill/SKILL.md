---
name: agent03
description: 跨境电商智能运营 Agent（Agent03）项目的开发与维护 Skill。用于对本项目进行代码维护、功能扩展、Bug 定位、业务理解、运行部署、接口调用。项目基于 LangGraph 显式状态机编排，覆盖竞品分析、商品信息抓取、Listing 文案生成、销售数据自然语言查询、知识库客服问答五大业务场景，并具备 BM25+Chroma 混合检索 RAG、FastAPI 服务封装、Docker 容器化、LangSmith 可观测等工程能力。触发词：Agent03、跨境电商运营助手、跨境智能运营、销售数据查询、竞品分析、Listing、客服问答、混合检索。
---

# 跨境电商智能运营 Agent（Agent03）维护开发 Skill

<!-- 文件作用：本文件是 skill 包唯一主控入口（SKILL.md），存放项目核心信息与硬约束，并索引 docs/ 下各参考文档。Agent 匹配到本 skill 后优先读取本文件；遇到具体深度任务时再按需读取对应 docs 文档（渐进式加载）。不修改任何现有业务代码。 -->

---

## 一、项目定位与能力边界

本系统是垂直领域智能运营 Agent，面向跨境电商运营人员，将四类重复性工作自动化：

| 业务场景 | task_type | 说明 |
| ---- | ---- | ---- |
| 竞品分析 | analysis | 基于公开图书站点 books.toscrape 抓取竞品，LLM 生成分析报告 |
| 商品信息抓取 | product_fetch | 传入商品 URL，抓取标题、价格、卖点并保存到知识库 |
| Listing 文案生成 | listing | 基于商品信息生成英文标题、五点描述、广告文案（三重校验） |
| 销售数据查询 | sales | 自然语言提取 SKU+月份，读取本地 CSV 返回结构化结果 |
| 知识库客服问答 | customer_service | 基于 RAG 召回结果 + LLM 生成客服回复 |

> **硬性约束（用户明确要求，开发时必须遵守）：**
> 1. 现阶段的业务逻辑和功能能力**不需要删减和优化**，禁止自主删减、改写或重构既有功能。
> 2. 新增能力/文档时不得修改现有代码文件的既有逻辑与格式。
> 3. 所有修改必须**基于现有代码**进行，返回完整修改后的代码，不提供片段。
> 4. 本项目为本地运行项目，依赖由 uv / Docker 管理，不使用手动虚拟环境。

---

## 二、技术栈与关键版本

- 语言/框架：Python ≥3.11、FastAPI、Pydantic v2
- Agent 编排：LangGraph（StateGraph 显式状态机 + MemorySaver checkpoint）
- LLM：LangChain + DeepSeek（`langchain-deepseek`，model=`deepseek-chat`）
- 检索：Chroma 向量库 + BM25（rank-bm25）+ RRF 融合 + BGE-reranker 重排序
- Embedding：BAAI/bge-m3（CPU，HF 国内镜像 hf-mirror.com）
- 部署：Docker（python:3.11-slim，uv 管理依赖）、uv
- 可观测：LangSmith（环境变量开关控制，非强依赖）

依赖清单以根目录 `pyproject.toml` 为准，所有软件包操作必须基于其中列出的依赖进行。

---

## 三、项目目录结构

```
Agent03/
├── main.py                  # FastAPI 主入口：lifespan 初始化、/api/agent/run、/health
├── pyproject.toml           # 项目依赖与构建配置（清华源）
├── Dockerfile               # 镜像构建（分层缓存）
├── uv.lock                  # 依赖锁定文件
├── .env                     # 环境变量（含 DEEPSEEK_API_KEY，已 gitignore）
├── .gitignore               # 忽略 .venv、__pycache__、构建产物
├── README.md                # 项目说明书
├── skill/
│   ├── SKILL.md             # 本主控文件
│   └── docs/                # 参考文档（渐进式加载）
│       ├── BUSINESS_FLOW.md # 完整业务流程与节点细则
│       ├── RAG_SYSTEM.md    # 混合检索体系详细说明
│       ├── API_REFERENCE.md # 接口规范与参数说明
│       └── DEPLOY_GUIDE.md  # 运行部署与 LangSmith 配置手册
├── agent/
│   ├── state.py             # AgentState / GraphState 状态定义
│   ├── graph_builder.py     # LangGraph 构图：节点注册、边、条件路由
│   ├── nodes.py             # 全部业务节点与路由函数
│   ├── tools.py             # 业务工具集（爬虫/竞品/Listing/销售/客服）
│   └── output_parsers.py    # LLM 输出解析与 Pydantic 校验
├── rag/
│   ├── build_vector_db.py   # 知识库加载、切分、向量库+BM25 构建
│   ├── hybrid_retriever.py  # 混合检索：BM25+Chroma+RRF+Rerank
│   ├── knowledge/           # 知识库文档（txt/md，UTF-8）
│   ├── sales_data.csv       # 销售数据
│   └── chroma_db/           # Chroma 向量库持久化目录
└── utils/
    └── helper.py            # trace_id、update_agent_state、文本/字典工具
```

---

## 四、核心架构与完整执行链路

### 4.1 状态定义（agent/state.py）
- `AgentState`：业务状态，所有业务字段的强类型容器。**关键约束：嵌套 Pydantic 对象禁止原地修改，必须用 `model_copy(update=...)` 整体替换返回。**
- `GraphState(MessagesState)`：LangGraph 顶层状态，`messages` 由原生 reducer 追加；`agent_state` 为嵌套业务对象，更新必须整体替换。
- 状态更新统一走 `utils/helper.update_agent_state()`，自动过滤非法字段并打印告警。

### 4.2 LangGraph 节点与链路（agent/graph_builder.py）
完整执行链路（基于代码真实路由）：

```
pre_rag_retrieve ──► query_rewrite ──► boundary_judge
                                         │
                    route_after_boundary │
              ┌──────────────────────────┴────────────┐
              │ hit_boundary=True                     │ hit_boundary=False
              ▼                                        ▼
        result_collect                           knowledge_judge
                                                   │
                              route_knowledge_judge│
                        ┌──────────────────────────┴────────────┐
                        │ need_retrieve=True                    │ need_retrieve=False
                        ▼                                        ▼
                   do_retrieve                            task_parser
                        │                                        │
                        └──────────► task_parser ◄────────────────┘
                                        │
                                     task_router
                              ┌──────────┴───────────┐
                              │ 有待执行/需重试        │ 全部完成
                              ▼                       ▼
                          exec_task             result_collect
                              │                        │
                              └──► task_router ────────► END
```

节点说明（均来自真实代码）：
- `pre_rag_retrieve_node`：入口。首轮初始化 AgentState，多轮则继承历史状态，更新本次 `user_query` 与 `trace_id`。
- `query_rewrite_node`：LLM 改写查询（补全模糊信息、不改变诉求），写入 `rewritten_query`，用于后续检索与意图判断。
- `boundary_judge_node`：边界判断。①硬关键词拦截（预测销量/下月销量/下个月销量/预估销量）→ `hit_boundary=True`；②同会话追问（历史 task_list 非空且 original_input≠query）→ 直接放行；③否则 LLM 判断知识库是否有明确兜底声明。
- `need_knowledge_judge_node`：LLM 判断是否需检索私有知识库，写 `need_retrieve` 与 `answer_source`（platform_knowledge / llm_knowledge）。
- `do_retrieve_node`：仅 `need_retrieve=True` 进入，调用 `hybrid_retriever.hybrid_search()`，top_k=8。
- `task_parser_node`：LLM 任务拆解，输出 `task_list` 与 `current_task_type`。**注意：只基于当前用户问题拆解，不传完整对话历史。**
- `exec_task_dispatcher`：按 `current_task_type` 分发到不同工具，失败写 `error_msg` 并 `retry_count+1`。
- `task_router`：条件路由（非节点）。重试达 `MAX_TASK_RETRY=2` 或全部任务完成 → `end`；有 error 未超限 → 重试；否则继续执行。
- `result_collect_node`：按 task_type 选取对应原始素材，交 LLM 润色成面向用户的最终回答，隐藏内部标记。

### 4.3 五大任务类型触发逻辑（task_parser_node 判定规则）
1. **customer_service**：商品 FAQ、参数说明、商品概况、爆款情况、笼统销售表现、纯闲聊/记忆类提问。
2. **analysis**：竞品对比、市场分析，拆分成多步子任务。
3. **listing**：需要商品标题、五点描述 Listing 文案。
4. **sales**：**必须同时出现 SKU/商品编号 AND 具体年月（如 2026-08）两个要素**；仅商品名称、无 SKU 或无月份，归 customer_service。
5. **product_fetch**：传入商品 URL，要求抓取商品信息。

---

## 五、关键开发红线（改动时不可违反）

1. **嵌套状态修改**：禁止 `agent_state.xxx = value`，必须 `model_copy(update=...)` 或 `update_agent_state(state, xxx=...)`。
2. **路由函数只返回字符串**：task_router、route_after_boundary、route_knowledge_judge 返回路由目标，不返回状态字典；普通节点才返回状态更新字典。
3. **task_parser 拆解**只基于当前用户问题（`msg_for_parse=[HumanMessage(content=user_query)]`），不传历史，避免历史长报告干扰解析。
4. **sales 判定**必须同时有 SKU 与年月，否则归 customer_service。
5. **竞品分析并发**：一次抓取共享商品数据，子任务仅并行做 LLM 分析，避免重复爬虫触发站点限流。
6. **listing 三重校验**：Prompt + PydanticOutputParser + `parse_listing_output` 业务规则校验（标题 60~120 字符全英文、五点固定 5 条、广告 ≤80 字符全英文），失败自动重试最多 2 次。
7. **result_collect 素材选取**：按 task_type 只选当前任务原始素材，不拼接全部；素材为空/空字典时走兜底文案，不送 LLM。
8. **接口并发重置**：每次 `/api/agent/run` 前置 `agent_graph.update_state(config, {"agent_state": clean_agent_state})` 清残留任务数据，但保留历史消息实现多轮记忆。
9. **LLM 输出解析**统一走 `safe_parse_json`（自动剥离 markdown 代码块），失败返回 None 并走兜底分支。
10. **重试熔断**：单个子任务最大重试 `MAX_TASK_RETRY=2`。
11. **BM25 索引仅内存态**：进程销毁即丢失，服务重启必须重新 `build_bm25_index()`；返回的 HybridRetriever 实例必须由主程序全程持有。
12. **HF 镜像前置**：`HF_ENDPOINT=https://hf-mirror.com` 必须放在所有 huggingface 相关 import 之前（build_vector_db.py 顶部已处理）。

---

## 六、附属文档索引（渐进式加载）

遇到以下深度任务时，按需读取对应文档，无需全部加载：

| 任务类型 | 对应文档 | 内容 |
| ---- | ---- | ---- |
| 理解业务流程、节点细节、状态流转、任务分发 | `docs/BUSINESS_FLOW.md` | 完整执行链路、五大任务判定、各节点读写、服务请求处理 |
| RAG 检索、知识库构建、混合检索调优 | `docs/RAG_SYSTEM.md` | BM25+Chroma+RRF+Rerank 细节、知识库构建流程、分词器、内存索引特性 |
| 接口调用、参数格式、错误码 | `docs/API_REFERENCE.md` | /api/agent/run 与 /health 规范、请求/响应体、错误码、lifespan |
| 本地运行、Docker 部署、LangSmith 配置 | `docs/DEPLOY_GUIDE.md` | 运行步骤、镜像构建、容器启动、运维命令、LangSmith 环境变量 |

---

## 七、快速排错指南

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| 接口报 402 Insufficient Balance | DeepSeek 账号余额不足 | 到 DeepSeek 开放平台充值，与代码无关 |
| docker run 报同名容器冲突 | `agent03_container` 已被占用 | `docker rm -f agent03_container` 后再 run |
| 启动阶段长时间卡住 | 首次加载 BGE-M3/reranker 需联网下载 | 属正常现象，等待；可挂载本地模型目录加速 |
| LangSmith 页面看不到 Trace | 缺少环境变量 / 项目名不匹配 | 确认 LANGSMITH_TRACING/ENDPOINT/API_KEY/PROJECT 四项，PROJECT 与平台一致 |
| 爬虫抓取失败/超时 | 站点限流 | 依赖既有重试、延时、并发控制逻辑，不擅改 |
| 外部无法访问接口 | 监听地址 / 端口映射不对 | FastAPI 监听 0.0.0.0，Docker 用 `-p 8000:8000` 映射 |

---

## 八、扩展指引（新增功能时遵循既有模式）

1. **新增业务场景**：在 `agent/tools.py` 实现工具函数并加入 `__all__` → 在 `agent/nodes.py` 的 `exec_task_dispatcher` 增加对应 task_type 分支 → 在 `task_parser_node` 的 prompt 增加判定规则 → 在 `result_collect_node` 增加素材选取分支 → 在 `agent/state.py` 增加结果字段。
2. **新增知识库文档**：将 UTF-8 编码的 txt/md 放入 `rag/knowledge/`，重启服务或调用构建函数即可入库。
3. **新增外部数据源**（如替换爬虫站点）：修改 `tools.py` 中的 URL 与页面选择器，保持返回 schema 不变。
4. **性能/稳定性**：遵循既有模式——重型资源全局单例、工具级异常捕获、重试熔断、降级兜底，不引入破坏性改动。

---

## 九、自测方式

各核心模块均内嵌 `if __name__ == "__main__"` 自测代码，可直接运行验证：
```
uv run python -m agent.state          # 状态模型自测
uv run python -m agent.graph_builder  # 构图与链路自测
uv run python -m agent.nodes          # 节点与路由自测
uv run python -m agent.tools          # 工具自测（默认 Mock 模式，爬虫块已注释）
uv run python -m agent.output_parsers # 解析校验自测
uv run python -m rag.hybrid_retriever # 混合检索自测
uv run python -m rag.build_vector_db  # 知识库构建自测
uv run python -m utils.helper         # 工具函数自测
```
> 注意：`agent.tools` 与 `rag.build_vector_db` 的自测会触发真实爬虫/模型加载，执行较慢，按需开启注释块。
