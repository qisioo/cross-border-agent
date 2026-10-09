<!-- 文件作用：本项目 RAG 混合检索体系参考文档，如实描述 BM25+Chroma+RRF+Rerank 检索流程、知识库构建流程、中文分词与内存索引特性。需要理解或调整检索/知识库时按需加载本文件。 -->

# Agent03 RAG 混合检索体系文档

本文档基于现有代码（rag/hybrid_retriever.py、rag/build_vector_db.py）如实描述系统真实检索与知识库构建机制。

---

## 1. 检索架构总览

采用「双路召回 + RRF 融合 + 重排序」的生产级混合检索链路，兼顾精确匹配与语义匹配。

```
用户查询（rewritten_query，由 query_rewrite 节点生成）
   │
   ├─► 向量检索  Chroma.similarity_search(top_k=8)
   ├─► 关键词检索 BM25Retriever.invoke（自定义中文分词器）
   │
   ▼
RRF 倒数排名融合（rrf_k=60）→ 按分数倒序
   │
   ▼
md5 去重
   │
   ▼
[重排序] FlagReranker(bge-reranker-v2-m3) 打分 → 过滤 <0.15 → 取 top_k
   │
   ▼
返回最终 Document 列表（存入 retrieved_chunks）
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
- 设置 `self.bm25_retriever.k = 3`。

### 2.2 Chroma 向量检索
- 使用 `langchain_chroma.Chroma`，持久化目录 `rag/chroma_db/`。
- Embedding：BAAI/bge-m3，`HuggingFaceEmbeddings`，CPU 运行，`normalize_embeddings=True`。

### 2.3 RRF 倒数排名融合
- 函数 `reciprocal_rank_fusion(vec_docs, bm25_docs, rrf_k=60)`。
- 两路结果按排名 `1/(rank + rrf_k)` 打分相加，同文档（按内容 md5 哈希识别）累加分数，无需人工调权重。

### 2.4 重排序
- 使用 `FlagEmbedding.FlagReranker`，模型 `BAAI/bge-reranker-v2-m3`，CPU 运行，`use_fp16=False`。
- 对 RRF 融合后的候选逐条打分，过滤 `rerank_threshold=0.15` 以下低分项，按分数倒序取 top_k。
- 加载失败（`load_reranker` 异常）时置 `self.reranker=None`，`hybrid_search` 自动降级为 RRF 结果。

---

## 3. 知识库构建流程（rag/build_vector_db.py）

```
读取 rag/knowledge/ 下 txt、md（UTF-8）
   → 过滤空文档、逐行清除行尾空白、压缩空行
   → RecursiveCharacterTextSplitter(chunk_size=220, overlap=45) 切分
     （separators=["\n\n","\n","。",". ","，",","," "]）
   → 文本内容去重（set 去重，保留首个）
   → [可选] 删除旧 chroma_db（full_rebuild=True）
   → 加载 BGE-M3 embedding（CPU）
   → 写入 Chroma 向量库（add_documents）
   → 构建内存 BM25 索引（build_bm25_index）
   → 返回 HybridRetriever 实例（必须全程持有）
```

### 3.1 构建入口 `build_vector_and_bm25(full_rebuild)`
- `full_rebuild=True`：先 `shutil.rmtree(CHROMA_DB_FOLDER)` 清空旧向量库，再全量重建，防止重复向量堆积。
- `full_rebuild=False`：增量追加；若向量库已有数据则跳过写入（不重复追加）。
- 加载 embedding 后写入向量库，再构建内存 BM25 索引，最后返回 `HybridRetriever` 实例。
- main.py 的 lifespan 中调用 `build_vector_and_bm25(full_rebuild=False)`，加载 reranker 后构建 Agent 图。

### 3.2 切分参数
- `chunk_size=220`，`chunk_overlap=45`，按 `\n\n、\n、。、. 、，、, 、空格` 分层切分。
- 切分后做内容去重（空 chunk 与重复 chunk 丢弃）。

---

## 4. 重要特性与注意事项（红线）

1. **BM25 索引仅内存态**：`build_bm25_index` 构建的索引只存在内存中，进程销毁即丢失；服务每次重启必须基于全部切片文档重新构建。
2. **retriever 实例必须全程持有**：返回的 `HybridRetriever` 实例若被丢弃，BM25 内存索引直接丢失。main.py 通过 lifespan 持有全局 `hybrid_retriever`。
3. **HF 镜像前置**：`HF_ENDPOINT=https://hf-mirror.com` 必须放在所有 huggingface 相关 import 之前（build_vector_db.py 顶部第 11 行已设置）。
4. **加载顺序**：`load_embedding()` 初始化 embedding 与 Chroma；`load_reranker()` 延迟加载重排序模型（可在构建后单独调用）。
5. **异常降级**：向量检索、BM25 检索、reranker 均带异常捕获，出错降级为空列表/降级结果，不中断 Agent 流程。

---

## 5. 检索相关字段

| 字段 | 位置 | 说明 |
| ---- | ---- | ---- |
| retrieved_chunks | agent/state.py AgentState | 存放 RAG 召回的知识库文本块，供边界判断与客服问答使用 |
| rewritten_query | agent/state.py AgentState | query_rewrite 节点写入，用于检索与意图判断 |
| answer_source | agent/state.py AgentState | 回答来源标记：platform_knowledge / llm_knowledge |

`do_retrieve_node` 调用 `hybrid_retriever.hybrid_search(rewritten_query, top_k=8)` 后将结果写入 `retrieved_chunks`。

---

## 6. 自测方式

```
uv run python -m rag.hybrid_retriever  # 混合检索自测（BM25 关键词 + hybrid_search）
uv run python -m rag.build_vector_db   # 知识库构建自测（默认仅测切分；取消注释可触发真实构建）
```
