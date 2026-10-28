<!-- 文件作用：本项目 RAG 混合检索体系参考文档，如实描述 BM25+Milvus+RRF+Rerank 检索流程、企业+对话级知识库构建流程、Qwen-VL 多模态识别、MD5 去重与内存索引特性。需要理解或调整检索/知识库时按需加载本文件。 -->

# Agent03 RAG 混合检索体系文档

本文档基于现有代码（rag/hybrid_retriever.py、rag/build_vector_db.py）如实描述系统真实检索与知识库构建机制。

---

## 1. 检索架构总览

采用「双路召回 + RRF 融合 + 重排序 + 动态 top-n」的生产级混合检索链路，兼顾精确匹配与语义匹配。

```
用户查询（子意图，由 intent_split 切分后传入 exec_intent）
   │
   ├─► 向量检索  Milvus.similarity_search(top_k=10，按企业/对话 collection)
   ├─► 关键词检索 BM25Retriever.invoke（自定义中文分词器，k=5）
   │
   ▼
RRF 倒数排名融合（rrf_k=60）→ 按分数倒序
   │
   ▼
md5 去重
   │
   ▼
[重排序] FlagReranker(bge-reranker-v2-m3) 打分 → 过滤 <0.15 → 动态 top-n
   │
   ▼
返回最终 Document 列表（存入客服问答的 retrieve_docs）
```

- 向量检索 / BM25 异常均降级返回空列表，不阻断链路。
- reranker 加载失败时优雅降级为 RRF 结果（不阻断流程）。

---

## 2. 核心组件说明

### 2.1 BM25 关键词检索
- 使用 `rank-bm25` 的 `BM25Retriever`。
- **自定义中文分词器 `chinese_tokenizer`**：中文按单字切分，连续英文/数字保留为整体 token。解决 BM25 默认空格分词对中文整句失效的问题，无需引入 jieba。
  ```python
  def chinese_tokenizer(text: str) -> List[str]:
      return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fa5]", text)
  ```
- 设置 `self.bm25_retriever.k = 5`（BM25 召回数量，关键词精确匹配少而精）。

### 2.2 Milvus 向量检索
- 使用 `langchain_milvus.Milvus`，连接本机 Milvus（`http://127.0.0.1:19530`，Docker standalone）。
- **多租户+对话级隔离**：企业级 collection `agent03_<tenant_id>`，对话级 collection `agent03_<tenant_id>_<conversation_id>`，`auto_id=True`；`full_rebuild=True` 时 `drop_old=True` 删除旧 collection 全量重建。
- Embedding：BAAI/bge-m3，`HuggingFaceEmbeddings`，CPU 运行，`normalize_embeddings=True`。

### 2.3 RRF 倒数排名融合
- 函数 `reciprocal_rank_fusion(vec_docs, bm25_docs, rrf_k=60)`。
- 两路结果按排名 `1/(rank + rrf_k)` 打分相加，同文档（按内容 md5 哈希识别）累加分数，无需人工调权重。

### 2.4 重排序 + 动态 top-n
- 使用 `FlagEmbedding.FlagReranker`，模型 `BAAI/bge-reranker-v2-m3`，CPU 运行，`use_fp16=False`。
- 对 RRF 融合后的候选逐条打分，过滤 `rerank_threshold=0.15` 以下低分项，按分数倒序。
- **动态 top-n（不是固定取前 N 个）**：遍历得分列表，找相邻得分差 ≥ `gap_threshold=0.15` 的断崖位置，断层之前的都保留；保底最多取 `top_k` 个（`hybrid_search` 默认 top_k=4）、最少取 1 个。
- 加载失败（`load_reranker` 异常）时置 `self.reranker=None`，`hybrid_search` 自动降级为 RRF 结果。

### 2.5 检索主入口 `hybrid_search(query, top_k=4)`
- 向量召回 `vector_search(query, top_k=10)` → BM25 召回 `keyword_search(query)` → RRF 融合 → md5 去重 → Rerank 过滤 + 动态 top-n → 返回。

---

## 3. 知识库构建流程（rag/build_vector_db.py）

### 3.1 文档加载与清洗
```
读取 知识库目录下的 txt、md、pdf（UTF-8）
   → 文件级 MD5 去重（相同内容的文件只处理一次）
   → 过滤空文档
   → 逐行清除行尾空白、压缩空行（伪空行转双换行、连续空行只留一组）
   → 生成 Document（metadata 含 source、file_md5）
```

### 3.2 PDF 多模态识别（Qwen-VL）
- **PDF 文本**：用 `pdfplumber` 提取每页文本。
- **PDF 图片**：用 PyMuPDF（fitz）提取页内图片 → 临时保存 → 调用 **Qwen-VL（`qwen3.8-omni-flash`，dashscope）** 识别图片内容（提取文字/表格/流程图结构）→ 追加到文本。
- 需 `DASHSCOPE_API_KEY`（从环境变量读取，不硬编码）；未配置则跳过图片识别，不报错。

### 3.3 切分
```
RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=45)
  separators=["\n\n","\n","。",". ","！","？","；",";","，",","," "]
   → 切分后文本块内容去重（空 chunk 与重复 chunk 丢弃）
```

### 3.4 三种构建入口
| 函数 | 作用 | collection / 目录 |
| ---- | ---- | ---- |
| `build_vector_and_bm25` | 通用构建（公共默认库） | `agent03_kb` / `rag/knowledge/` |
| `build_tenant_retriever` | 企业级多租户构建 | `agent03_<tenant_id>` / `rag/knowledge/<tenant_id>/` |
| `build_conversation_retriever` | 对话级隔离构建 | `agent03_<tenant_id>_<conversation_id>` / `rag/knowledge/<tenant_id>/<conversation_id>/` |

- 均支持 `full_rebuild`（True 删旧重建防向量堆积；False 复用已有 collection，row_count>0 则跳过写入）与 `progress_callback`（进度 0-100）。
- 构建完成后注册到 `TenantRetrieverRegistry`（企业级 key=`<tenant_id>`，对话级 key=`<tenant_id>::<conversation_id>`）。

---

## 4. 重要特性与注意事项（红线）

1. **BM25 索引仅内存态**：`build_bm25_index` 构建的索引只存在内存中，进程销毁即丢失；服务每次重启必须重新触发知识库构建/重建（上传或 rebuild 接口）。
2. **retriever 实例必须全程持有**：返回的 `HybridRetriever` 实例若被丢弃，BM25 内存索引直接丢失。多租户/对话级下由 `TenantRetrieverRegistry` 长期持有。
3. **HF 镜像前置**：`HF_ENDPOINT=https://hf-mirror.com` 必须放在所有 huggingface 相关 import 之前（build_vector_db.py 顶部已设置）。
4. **加载顺序**：`load_embedding()` 初始化 embedding 并连接/创建 Milvus collection；`load_reranker()` 延迟加载重排序模型（构建时显式调用，全局共享）。
5. **异常降级**：向量检索、BM25 检索、reranker 均带异常捕获，出错降级为空列表/降级结果，不中断 Agent 流程。
6. **对话级隔离**：检索只查本对话绑定的 collection（`agent03_<tenant>_<conv>`），绝不跨对话；该对话未构建知识库时返回空检索。

---

## 5. 检索相关字段

| 字段 | 位置 | 说明 |
| ---- | ---- | ---- |
| intent_list | agent/state.py AgentState | 切分后的原子子意图（含 type/business/group_id） |
| sub_results | agent/state.py AgentState | 各子意图执行结果（客服子意图含 RAG 召回后的回答） |
| tenant_id / conversation_id | agent/state.py AgentState | 企业/对话标识，用于从注册表取对应检索器 |

`exec_intent_node` 按 `state.tenant_id` + `state.conversation_id` 从租户注册表取对应检索器，调用 `hybrid_search(子意图, top_k=4)` 后将召回文档传给 `customer_service_qa`。

---

## 6. 自测方式

```
uv run python -m rag.hybrid_retriever  # 混合检索自测（BM25 关键词 + hybrid_search）
uv run python -m rag.build_vector_db   # 知识库构建自测（默认仅测切分；取消注释可触发真实构建）
```
