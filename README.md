# 跨境电商智能运营 Agent（Agent03）

面向跨境电商运营场景的垂直领域智能 Agent。基于 LangGraph 显式状态机编排，将竞品分析、商品信息抓取、Listing 文案生成、销售数据自然语言查询、知识库客服问答五大重复性工作自动化，配合 BM25+Milvus 混合检索 RAG（多租户企业知识库隔离）与 FastAPI 服务封装，配套 Streamlit 前端工作台，支持 Docker 一键部署与 LangSmith 全链路可观测。

---

## 1. 功能概览

| 业务场景 | 说明 |
| ---- | ---- |
| 竞品分析 | 基于公开图书站点抓取竞品，LLM 生成竞品分析报告 |
| 商品信息抓取 | 传入商品 URL，抓取标题、价格、卖点，保存到知识库 |
| Listing 文案生成 | 基于商品信息生成英文标题、五点描述、广告文案（三重校验） |
| 销售数据查询 | 自然语言提取 SKU+月份，读取本地 CSV 返回结构化销售数据 |
| 知识库客服问答 | 基于 RAG 混合检索 + LLM 生成客服回复，禁止编造 |

---

## 2. 技术栈

- **语言/框架**：Python ≥3.11、FastAPI、Pydantic v2
- **Agent 编排**：LangGraph（StateGraph 显式状态机 + MemorySaver checkpoint）
- **LLM**：LangChain + DeepSeek（`deepseek-chat`）
- **检索**：Milvus 向量库（多租户 `agent03_<tenant_id>` collection）+ BM25 + RRF 融合 + BGE-reranker 重排序
- **Embedding**：BAAI/bge-m3（CPU，HF 国内镜像）
- **部署**：Docker（python:3.11-slim + uv）、uv 依赖管理
- **可观测**：LangSmith（环境变量开关，非强依赖）

完整依赖见 `pyproject.toml`。

---

## 3. 目录结构

```
Agent03/
├── main.py                  # FastAPI 主入口
├── streamlit_app.py         # Streamlit 前端工作台
├── pyproject.toml           # 依赖与构建配置
├── Dockerfile               # 镜像构建
├── uv.lock                  # 依赖锁定
├── .env                     # 环境变量（含 DeepSeek API Key）
├── SKILL.md                 # 项目维护开发 Skill
├── README.md                # 本文件（项目说明书）
├── docs/
│   └── BUSINESS_FLOW.md     # 完整业务流程文档
├── agent/
│   ├── state.py             # 状态定义
│   ├── graph_builder.py     # LangGraph 构图
│   ├── nodes.py             # 业务节点与路由
│   ├── tools.py             # 业务工具集
│   └── output_parsers.py    # LLM 输出解析与校验
├── rag/
│   ├── build_vector_db.py   # 知识库构建
│   ├── hybrid_retriever.py  # 混合检索
│   ├── knowledge/           # 知识库文档（按企业子目录隔离）
│   ├── tenant_registry.py   # 租户检索注册表（多租户隔离）
│   ├── sales_data.csv       # 销售数据
└── utils/
    └── helper.py            # 通用工具函数
```

---

## 4. 快速开始

### 4.1 环境准备
1. 安装 Python 3.11+ 与 uv。
2. 配置 `.env`（参考现有 `.env`）：
   ```
   DEEPSEEK_API_KEY=你的DeepSeek密钥
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   ```

### 4.2 安装依赖
```
uv sync
```
> 依赖安装走清华 PyPI 源（已在 `pyproject.toml` 配置）。本项目本地运行，不额外创建虚拟环境。

### 4.3 启动服务
```
uv run python -m main
```
启动后访问：
- Swagger 调试页：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

> 服务启动时创建多租户检索注册表（纯隔离，不构建公共默认库）。每个企业的知识库由上传接口按 tenant_id 构建进 Milvus（collection `agent03_<tenant_id>`），BGE-M3 embedding 首次加载耗时较长属正常现象。

---

## 5. 接口说明

### POST `/api/agent/run`
调用 Agent 处理一条用户请求。

**请求体**
```json
{
  "user_query": "查询SKU-B002在2026-08的销售数据",
  "session_id": "sess-001",
  "tenant_id": "enterprise_a"
}
```
- `user_query`：必填，用户提问，长度 ≤800 字符。
- `session_id`：选填，多轮会话标识。
- `tenant_id`：选填，企业/租户标识（默认 `default`）。多租户知识库隔离：对话线程以 `tenant_id:session_id` 隔离，检索只用该企业的知识库。

**响应体**
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
- `final_answer`：面向用户的最终回答。
- `data`：后台调试用的业务状态字段（已过滤内部标记）。

**错误码**
| 状态码 | 含义 |
| ---- | ---- |
| 400 | 输入为空 / 超长 |
| 503 | Agent 服务尚未初始化完成 |
| 504 | Agent 执行超时（默认 60s） |
| 500 | 服务内部异常 |

### GET `/health`
探活接口，返回 `{"status": "ok"}` 表示服务已就绪。

### POST `/api/tenant/upload`
上传企业知识库文档（txt/md/csv），按 `tenant_id` 异步构建该企业独立向量库（Milvus collection `agent03_<tenant_id>`）与 BM25 索引，返回 `task_id` 供进度查询。**纯隔离**：各企业知识库互不共享。

### GET `/api/tenant/build/{task_id}`
查询企业知识库构建进度，返回 `{status, progress, message}`，前端用于进度条轮询（"正在构建向量数据库"）。

---

## 6. Docker 部署

### 6.1 构建镜像
```
docker build -t cross-border-agent:v1 .
```

### 6.2 启动容器
```
docker run -d -p 8000:8000 --name agent03_container \
  -e DEEPSEEK_API_KEY="你的DeepSeek密钥" \
  cross-border-agent:v1
```

### 6.3 开启 LangSmith 追踪（可选）
追加环境变量：
```
  -e LANGSMITH_TRACING=true \
  -e LANGSMITH_ENDPOINT=https://api.smith.langchain.com \
  -e LANGSMITH_API_KEY="你的LangSmith密钥" \
  -e LANGSMITH_PROJECT="Agent03"
```

### 6.4 常用运维
```
docker ps                        # 查看容器状态
docker logs -f agent03_container # 查看实时日志
docker rm -f agent03_container   # 删除容器（不影响镜像）
```
> 修改代码后需重新 build 镜像、删除旧容器再 run（同名容器会冲突）。Docker Desktop 图形界面可点「暂停/播放」启停容器，删除容器后需重新 run。

---

## 7. 知识库管理

- **多租户纯隔离**：每个企业的知识库独立存放于 `rag/knowledge/<tenant_id>/`，独立构建 Milvus collection（`agent03_<tenant_id>`）与内存 BM25 索引，企业间互不污染。
- 通过前端工作台（`streamlit_app.py`）或 `POST /api/tenant/upload` 上传企业文档（txt/md/csv），后端异步构建并返回进度。
- 销售数据存放于 `rag/sales_data.csv`，字段：`product_id, month, sales_volume, revenue, conversion_rate`。

---

## 8. 开发与调试

- 各核心模块内嵌 `if __name__ == "__main__"` 自测代码，可直接运行：
  ```
  uv run python -m agent.state
  uv run python -m agent.graph_builder
  uv run python -m agent.nodes
  uv run python -m agent.tools
  uv run python -m agent.output_parsers
  uv run python -m rag.hybrid_retriever
  uv run python -m rag.build_vector_db
  uv run python -m utils.helper
  ```
- 本项目维护开发规范详见 `SKILL.md`，完整业务流程详见 `docs/BUSINESS_FLOW.md`。
- 项目已接入 LangSmith：调用接口后可在 LangSmith 平台（项目名 `Agent03`）查看每个节点的输入输出、耗时、Token 消耗与异常。

---

## 9. 维护约束（重要）

1. **不删减、不优化既有功能**：现阶段的业务逻辑与功能能力保持现状，仅可新增能力或修复 Bug。
2. **修改基于现有代码**：返回完整修改后的代码，不提供片段。
3. **状态修改红线**：嵌套 `agent_state` 禁止原地修改，必须 `model_copy(update=...)` 或走 `update_agent_state()`。
4. **依赖约束**：软件包操作基于 `pyproject.toml` 现有依赖进行。
5. **敏感信息**：`.env` 含密钥，已加入 `.gitignore`，上传 Git 时勿提交；代码中不使用硬编码密钥，统一从环境变量读取。
