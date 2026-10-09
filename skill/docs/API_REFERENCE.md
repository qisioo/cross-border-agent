<!-- 文件作用：本项目 FastAPI 接口参考文档，如实描述 /api/agent/run 与 /health 接口的请求/响应规范、错误码、生命周期初始化机制。需要调用接口或理解服务入口时按需加载本文件。 -->

# Agent03 API 参考文档

本文档基于现有代码（main.py）如实描述系统真实 HTTP 接口规范。

---

## 1. 服务入口（main.py）

- FastAPI 应用：`app = FastAPI(title="Agent03智能体服务", lifespan=lifespan)`。
- 启动方式：`uvicorn main:app --host 0.0.0.0 --port 8000`（绑定所有网卡供外部/容器访问）。
- Swagger 调试页：`http://127.0.0.1:8000/docs`。

### 1.1 lifespan 生命周期初始化
服务启动时一次性完成重型资源初始化，接口层只做推理：
1. `build_vector_and_bm25(full_rebuild=False)` 构建/加载向量库与 BM25 索引。
2. `hybrid_retriever.load_reranker()` 加载 BGE-reranker 重排序模型。
3. `build_graph(hybrid_retriever)` 构建并编译 LangGraph Agent 图，存入全局 `agent_graph`。
4. 初始化完成前访问 `/api/agent/run` 返回 503。

> 首启会加载 BGE-M3 embedding 与 reranker，耗时较长属正常现象。

---

## 2. POST /api/agent/run

调用 Agent 处理一条用户请求。

### 2.1 请求体（AgentRequest）
```json
{
  "user_query": "查询SKU-B002在2026-08的销售数据",
  "session_id": "sess-001"
}
```
| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | ---- | ---- |
| user_query | str | 是 | 用户提问，去空格后长度 ≤800 字符（MAX_QUERY_LENGTH） |
| session_id | str | 否 | 多轮会话标识，默认 `"sess-default-001"` |

### 2.2 响应体
```json
{
  "code": 0,
  "msg": "success",
  "trace_id": "trace-xxx",
  "cost_seconds": 3.14,
  "final_answer": "【基于电商平台知识库回答】……",
  "data": { }
}
```
| 字段 | 说明 |
| ---- | ---- |
| code / msg | 业务状态码与消息，`0` / `success` 表示成功 |
| trace_id | 会话追踪 ID，用于日志与 LangSmith 排查 |
| cost_seconds | Agent 执行耗时（秒） |
| final_answer | 面向用户的最终回答（已正则清洗子任务标记） |
| data | 后台调试用的业务状态字段（已过滤内部标记字段） |

`final_answer` 提取顺序：`agent_state.final_answer` → `competitor_report` → 最后一条 AIMessage。

### 2.3 过滤字段（FILTER_OUT_FIELDS，不返回在 data 中）
`trace_id`、`retry_count`、`current_task_index`、`task_list`、`error_msg`、`original_input`。

### 2.4 会话重置逻辑（多轮记忆关键）
每次请求执行前：
1. `agent_graph.get_state(config)` 读取历史消息。
2. 拼接历史 + 本次 user 消息作为输入 payload。
3. `agent_graph.update_state(config, {"agent_state": clean_agent_state})` 用全新的 AgentState 覆盖 checkpoint 中残留的历史任务数据，但保留历史消息，实现多轮对话记忆的同时避免历史任务状态干扰。

---

## 3. GET /health

探活接口：
```json
// agent_graph 已初始化
{"status": "ok"}
// agent_graph 未初始化
{"status": "unavailable", "reason": "agent graph not initialized"}
```

---

## 4. 错误码

| HTTP 状态码 | 含义 | 触发条件 |
| ---- | ---- | ---- |
| 400 | 输入非法 | `user_query` 为空，或长度超过 800 字符 |
| 503 | 服务未就绪 | `agent_graph is None`，lifespan 尚未完成初始化 |
| 504 | 执行超时 | `AGENT_RUN_TIMEOUT=60` 秒内未完成，竞品分析类任务易触发 |
| 500 | 服务内部异常 | Agent 执行过程中抛出未捕获异常 |

> 大模型服务商（如 DeepSeek）返回 402 Insufficient Balance 属账号余额问题，会以 500 形式返回，需充值解决。

---

## 5. 主要配置常量（main.py）

| 常量 | 值 | 说明 |
| ---- | ---- | ---- |
| MAX_QUERY_LENGTH | 800 | 用户输入最大字符数 |
| AGENT_RUN_TIMEOUT | 60 | Agent 执行超时（秒） |
| FILTER_OUT_FIELDS | 集合 | 响应 data 中过滤的内部字段 |

---

## 6. 自测方式

```
uv run python -m main   # 启动服务，访问 /docs 与 /health 验证
```
