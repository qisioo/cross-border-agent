<!-- 文件作用：本项目本地运行与部署运维参考文档，如实描述 uv 依赖管理、服务启动、外部依赖（Milvus/MCP）、Docker 镜像构建与容器部署、LangSmith 配置、常用运维命令。需要部署或运维时按需加载本文件。 -->

# Agent03 运行部署与运维文档

本文档基于现有配置（pyproject.toml、Dockerfile、.env）如实描述系统真实的本地运行与 Docker 部署方式。

---

## 1. 本地运行

### 1.1 环境准备
1. 安装 Python 3.11+ 与 uv。
2. 启动本地依赖服务：
   - **Milvus**（Docker standalone，端口 19530）：向量库。
   - **MCP Server**（`mcp_demo/server.py`）：提供 get_weather/get_time，需用 `mcp_demo` 的 .venv 解释器启动。
3. 配置根目录 `.env`（参考现有 `.env`）：
   ```
   DEEPSEEK_API_KEY=你的DeepSeek密钥
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   DASHSCOPE_API_KEY=你的DashScope密钥   # Qwen-VL PDF图片识别
   ```

### 1.2 安装依赖
```
uv sync
```
- 依赖走清华 PyPI 源（`pyproject.toml` 中 `[[tool.uv.index]]` 已配置 `https://pypi.tuna.tsinghua.edu.cn/simple`）。
- 本项目本地运行，不额外创建手动虚拟环境；依赖由 uv 管理。

### 1.3 启动服务
```
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```
- 服务监听 `0.0.0.0:8000`。
- Swagger 调试页：`http://127.0.0.1:8000/docs`。
- 健康检查：`http://127.0.0.1:8000/health`。

> 首启 lifespan 只预加载 Skill 规则，不加载 embedding/reranker（延迟到知识库构建时）。

---

## 2. 外部依赖

### 2.1 Milvus（向量库）
- 连接地址：`http://127.0.0.1:19530`（rag/hybrid_retriever.py 的 `MILVUS_URI`）。
- 采用 Docker standalone 部署，需保持端口 19530 可达。
- 各企业/对话知识库按需构建 collection `agent03_<tenant>[_<conv>]`。

### 2.2 MCP Server（天气/时间工具）
- MCP Server 位于 `D:\qi_agent\mcp_demo\server.py`，提供 `get_weather` / `get_time`。
- 必须用 `mcp_demo` 的 .venv 解释器（`mcp_demo\.venv\Scripts\python.exe`）启动，否则找不到 mcp 库。
- Agent 侧通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 以 stdio 方式连接，工具实现与 Agent 解耦，可复用。

---

## 3. Docker 部署

### 3.1 Dockerfile 结构（分层构建，缓存复用）
```
FROM python:3.11-slim                 # 基础镜像
WORKDIR /app
RUN apt-get install git                # 系统依赖
RUN pip install uv (清华源, 超时120s)   # 包管理工具
COPY pyproject.toml uv.lock ./         # 依赖清单
RUN uv sync --frozen (清华源)           # 安装依赖
COPY . .                               # 业务代码
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
设计原则：不变的层放下面、易变的层放上面，最大化缓存复用；只改业务代码时重新构建只跑最后几层。

### 3.2 构建镜像
```
docker build -t cross-border-agent:v1 .
```

### 3.3 启动容器
```
docker run -d -p 8000:8000 --name agent03_container \
  -e DEEPSEEK_API_KEY="你的DeepSeek密钥" \
  -e DASHSCOPE_API_KEY="你的DashScope密钥" \
  -e MILVUS_URI="http://host.docker.internal:19530" \
  cross-border-agent:v1
```
> Milvus 在宿主机时，容器内需用 `host.docker.internal` 访问宿主机端口。

### 3.4 开启 LangSmith 追踪（可选）
在启动命令追加环境变量：
```
  -e LANGSMITH_TRACING=true \
  -e LANGSMITH_ENDPOINT=https://api.smith.langchain.com \
  -e LANGSMITH_API_KEY="你的LangSmith密钥" \
  -e LANGSMITH_PROJECT="Agent03"
```
- LangSmith 是可选可观测组件，不配置不影响核心业务运行（`@traceable` 自动静默降级，不报错）。
- 需要 4 个变量齐全、且 PROJECT 与 LangSmith 平台项目名一致才会上报。

---

## 4. 常用运维命令

```
docker ps                        # 查看容器状态
docker logs -f agent03_container # 查看实时日志
docker rm -f agent03_container   # 删除容器（不影响镜像）
```

> **重要**：
> 1. 修改代码后需重新 `docker build` 构建新镜像、删除旧容器再 `docker run`（同名容器会冲突，报 `Conflict. The container name ... already in use`）。
> 2. `docker rm -f` 仅删除容器实例，不影响镜像 `cross-border-agent:v1`。
> 3. Docker Desktop 图形界面可点「暂停/播放」启停容器；但容器一旦 run 起来环境变量已固化，新增/修改环境变量必须删除容器重新 run。
> 4. 容器内首次加载 BGE-M3/reranker 需联网（HF 国内镜像 hf-mirror.com）；可挂载本地模型目录加速启动。

---

## 5. 环境变量汇总

| 变量 | 必填 | 说明 |
| ---- | ---- | ---- |
| DEEPSEEK_API_KEY | 是 | DeepSeek 模型密钥 |
| DEEPSEEK_BASE_URL | 是 | DeepSeek 接口地址 |
| DASHSCOPE_API_KEY | 条件 | Qwen-VL PDF 图片识别（不配置则跳过图片识别） |
| MILVUS_URI | 条件 | Milvus 连接地址（默认 http://127.0.0.1:19530） |
| LANGSMITH_TRACING | 可选 | true/false，是否上报追踪 |
| LANGSMITH_ENDPOINT | 可选 | LangSmith 上报地址 |
| LANGSMITH_API_KEY | 可选 | LangSmith 密钥 |
| LANGSMITH_PROJECT | 可选 | LangSmith 项目名（如 Agent03） |

> 项目相关 API key（DEEPSEEK_API_KEY、DASHSCOPE_API_KEY 等）统一从 `.env` 环境获取，代码不硬编码；`.env` 已加入 `.gitignore`，上传 Git 时勿提交。
