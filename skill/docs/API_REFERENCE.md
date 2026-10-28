<!-- 文件作用：本项目 FastAPI 接口参考文档，如实描述多租户/对话级知识库管理接口、/api/agent/run、/api/agent/stream(SSE)、/health 的请求/响应规范、错误码、生命周期初始化机制。需要调用接口或理解服务入口时按需加载本文件。 -->

# Agent03 API 参考文档

本文档基于现有代码（main.py）如实描述系统真实 HTTP 接口规范。

---

## 1. 服务入口（main.py）

- FastAPI 应用：`app = FastAPI(title="Agent03智能体服务", lifespan=lifespan)`。
- 启动方式：`uv run uvicorn main:app --host 0.0.0.0 --port 8000`（绑定所有网卡供外部/容器访问）。
- Swagger 调试页：`http://127.0.0.1:8000/docs`。

### 1.1 lifespan 生命周期初始化
服务启动时一次性完成重型资源初始化：
1. `get_skill_loader()` 预加载 skill/rules 全部分片（渐进式加载 Level1）。
2. `get_tenant_registry()` 创建租户检索注册表（多租户纯隔离，不构建公共默认库；各企业/对话知识库由上传/rebuild 接口单独构建注册）。
3. `build_graph(tenant_registry)` 构建并编译 LangGraph Agent 图，存入全局 `agent_graph`。
4. 初始化完成前访问 `/api/agent/run`、`/api/agent/stream` 返回 503。

> 首启会加载 Skill 规则，不加载 embedding/reranker（延迟到知识库构建时）。

---

## 2. 多租户知识库接口

### 2.1 POST /api/tenant/upload
上传企业知识库文档，异步构建企业独立向量库（`agent03_<tenant_id>`）+ BM25 索引。
- 请求：`multipart/form-data`，字段 `tenant_id` + `file`（支持 txt/md/csv/pdf）。
- 校验：tenant_id 仅字母/数字/下划线/连字符，长度 1-64；文件仅 txt/md/csv/pdf。
- 响应：`{"task_id","tenant_id","status":"running","message"}`。
- 构建在后台线程执行，用 task_id 查询进度。

### 2.2 GET /api/tenant/build/{task_id}
查询知识库构建进度，前端用于进度条轮询。
- 响应：`{"task_id","tenant_id","status","progress"(0-100),"message"}`。

---

## 3. 对话级知识库接口（企业主ID下的多对话）

### 3.1 对话 CRUD
| 方法 | 路径 | 作用 |
| ---- | ---- | ---- |
| POST | `/api/tenant/{tenant_id}/conversations` | 创建对话，body `{"name": "对话N"}`，返回对话元数据 |
| GET | `/api/tenant/{tenant_id}/conversations` | 列出某企业全部对话 |
| PUT | `/api/tenant/{tenant_id}/conversations/{conversation_id}` | 改对话显示名，body `{"name": ""}` |
| DELETE | `/api/tenant/{tenant_id}/conversations/{conversation_id}` | 删除对话（清元数据+文件目录+Milvus向量库+注册表） |
| GET | `/api/tenant/{tenant_id}/conversations/{conversation_id}` | 对话详情（元数据+知识库文件列表+构建状态） |

### 3.2 对话知识库管理
| 方法 | 路径 | 作用 |
| ---- | ---- | ---- |
| POST | `/api/tenant/{tenant_id}/conversations/{conversation_id}/upload` | 上传对话知识库文档，异步构建 `agent03_<tenant>_<conv>` |
| GET | `/api/tenant/{tenant_id}/conversations/{conversation_id}/knowledge` | 列出对话知识库文件 + 构建状态（chunk 数） |
| DELETE | `/api/tenant/{tenant_id}/conversations/{conversation_id}/knowledge/{filename}` | 删除对话下某个知识库文件，并触发重建 |
| POST | `/api/tenant/{tenant_id}/conversations/{conversation_id}/rebuild` | 手动重建对话知识库（重新索引现有全部文件） |

> 校验：tenant_id 同企业接口；conversation_id 仅字母/数字（uuid hex），长度 1-64。对话不存在返回 404。

---

## 4. POST /api/agent/run

调用 Agent 处理一条用户请求。

### 4.1 请求体（AgentRequest）
```json
{
  "user_query": "查询SKU-B002在2026-08的销售数据",
  "tenant_id": "default",
  "conversation_id": "8c6fd7b3",
  "session_id": "sess-001"
}
```
| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | ---- | ---- |
| user_query | str | 是 | 用户提问，去空格后长度 ≤800 字符（MAX_QUERY_LENGTH） |
| tenant_id | str | 否 | 企业/租户标识，默认 `"default"` |
| conversation_id | str | 否 | 对话标识，与 tenant_id 共同实现对话级隔离 |
| session_id | str | 否 | 会话标识，默认 `"sess-default-001"` |

> `thread_id = f"{tenant_id}:{conversation_id}:{session_id}"`（有 conversation_id）或 `f"{tenant_id}:{session_id}"`，用于 LangGraph checkpoint 隔离不同企业/不同对话的历史。

### 4.2 响应体
```json
{
  "code": 0,
  "msg": "success",
  "trace_id": "trace-xxx",
  "cost_seconds": 3.14,
  "final_answer": "……",
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

### 4.3 过滤字段（FILTER_OUT_FIELDS，不返回在 data 中）
`trace_id`、`retry_count`、`current_task_index`、`task_list`、`error_msg`、`original_input`。

### 4.4 会话重置逻辑（多轮记忆关键）
每次请求执行前：
1. `agent_graph.update_state(config, {"agent_state": clean_agent_state})` 用全新的 AgentState 覆盖 checkpoint 中残留的历史任务数据。
2. 只传本次 user 消息（`payload={"messages":[HumanMessage(...)]}`），不手动拼历史——LangGraph checkpoint 自动恢复历史 messages，`add_messages` reducer 只追加新消息。
3. 实现多轮对话记忆的同时避免历史任务状态干扰。

---

## 5. POST /api/agent/stream（SSE 流式）

LangGraph 原生流式 + 模拟打字机推送。

### 5.1 请求体
与 `/api/agent/run` 相同（AgentRequest）。

### 5.2 响应（`text/event-stream`）
事件序列（`data: {json}`）：
| type | 说明 |
| ---- | ---- |
| start | 起始事件，含 trace_id |
| node | 节点执行提示（`node` + `hint` 中文提示），如 intent_split→"正在切分意图..." |
| delta | token 级回答片段（step=4 字符切分，0.02s 间隔，模拟打字机） |
| done | 结束事件，含 trace_id |
| error | 异常/超时消息 |

节点到中文提示映射（NODE_NAME_MAP）：
`pre_rag_retrieve→正在前置检索... / intent_split→正在切分意图... / reflection_check→正在反思校验... / intent_execute→正在执行任务... / result_collect→正在汇总结果... / output_reflection→正在输出校验...`

---

## 6. GET /health

探活接口：
```json
// agent_graph 已初始化
{"status": "ok"}
// agent_graph 未初始化
{"status": "unavailable", "reason": "agent graph not initialized"}
```

---

## 7. 错误码

| HTTP 状态码 | 含义 | 触发条件 |
| ---- | ---- | ---- |
| 400 | 输入非法 | user_query 为空/超长，或 tenant_id/conversation_id 格式非法，或文件类型不支持 |
| 404 | 资源不存在 | 对话不存在 / 知识库文件不存在 / 构建任务不存在 |
| 503 | 服务未就绪 | `agent_graph is None`，lifespan 尚未完成初始化 |
| 504 | 执行超时 | `AGENT_RUN_TIMEOUT=60` 秒内未完成，竞品分析类任务易触发 |
| 500 | 服务内部异常 | Agent 执行过程中抛出未捕获异常 |

> 大模型服务商（如 DeepSeek）返回 402 Insufficient Balance 属账号余额问题，会以 500 形式返回，需充值解决。

---

## 8. 主要配置常量（main.py）

| 常量 | 值 | 说明 |
| ---- | ---- | ---- |
| MAX_QUERY_LENGTH | 800 | 用户输入最大字符数 |
| AGENT_RUN_TIMEOUT | 60 | Agent 执行超时（秒） |
| FILTER_OUT_FIELDS | 集合 | 响应 data 中过滤的内部字段 |
| MILVUS_URI | http://127.0.0.1:19530 | Milvus 连接地址（rag/hybrid_retriever.py） |

---

## 9. 自测方式

```
uv run uvicorn main:app --host 0.0.0.0 --port 8000   # 启动服务
# 访问 /docs 与 /health 验证
```
