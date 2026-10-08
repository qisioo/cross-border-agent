""" 混合检索：BM25关键词检索 + Chroma向量检索 RRF倒数排名融合 + Rerank重排序
生产级RAG召回 面试重点：RRF融合、异常降级、BM25内存索引特性、重排序降噪
BM25索引仅内存生效，进程销毁索引丢失；程序启动必须传入全部切片文档重建build_bm25_index
"""
from typing import List, Tuple, Optional, Dict
import hashlib
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def chinese_tokenizer(text: str) -> List[str]:
    """
    中文分词器：中文按单字切分，连续英文/数字保留为整体token
    解决BM25默认空格分词对中文整句失效的问题，无需引入jieba
    :param text: 原始文本
    :return: token列表
    """
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fa5]", text)


def calc_doc_hash(doc: Document) -> str:
    """生成文档片段哈希，用于去重，替代直接对比完整文本"""
    content = doc.page_content.strip()
    return hashlib.md5(content.encode("utf‑8")).hexdigest()


def reciprocal_rank_fusion(
    vec_docs: List[Document],
    bm25_docs: List[Document],
    rrf_k: int = 60
) -> List[Tuple[float, Document]]:
    """
    RRF 倒数排名融合算法
    :param vec_docs: 向量检索结果，顺序为返回排名（0是第1名 rank=1）
    :param bm25_docs: BM25检索结果
    :param rrf_k: RRF平滑系数，行业常用60
    :return: (rrf_score, document) 列表，未排序
    """
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # 向量检索打分
    for rank_idx, doc in enumerate(vec_docs):
        rank = rank_idx + 1
        h = calc_doc_hash(doc)
        score = 1.0 / (rank + rrf_k)
        score_map[h] = score_map.get(h, 0.0) + score
        doc_map[h] = doc

    # BM25检索打分
    for rank_idx, doc in enumerate(bm25_docs):
        rank = rank_idx + 1
        h = calc_doc_hash(doc)
        score = 1.0 / (rank + rrf_k)
        score_map[h] = score_map.get(h, 0.0) + score
        doc_map[h] = doc

    result = [(score_map[h], doc_map[h]) for h in score_map.keys()]
    return result


class HybridRetriever:
    def __init__(self, persist_directory: str = "./rag/chroma_db"):
        self.embedding_func = None
        self.persist_dir = persist_directory
        self.vector_store: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        # 使用BGE官方FlagReranker
        self.reranker = None
        self.rerank_threshold = 0.15  # 重排序过滤阈值，低于该分数直接丢弃

    def load_embedding(self):
        """延迟加载本地BGE‑M3 embedding模型，真实业务调用才初始化；自测不会加载大模型"""
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedding_func = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_func
        )

    def load_reranker(self):
        """延迟加载官方BGE‑reranker‑v2‑m3，CPU运行"""
        if self.reranker is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=False
            )
            print("[HybridRetriever] FlagReranker 重排序模型加载成功")
        except Exception as e:
            print(f"[HybridRetriever] reranker加载失败: {str(e)}，跳过重排序，降级使用RRF+去重")
            self.reranker = None

    def build_bm25_index(self, docs: List[Document]) -> None:
        """
        根据文档集合构建BM25内存索引
        ⚠️重要：索引仅保存在内存，程序重启消失；每次启动需要传入全部切片文档重建
        """
        # 传入中文分词器，修复中文整句无法匹配的问题
        self.bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=chinese_tokenizer
        )
        self.bm25_retriever.k = 3

    def vector_search(self, query: str, top_k: int = 3) -> List[Document]:
        """向量相似度检索，增加异常捕获，出错返回空列表降级"""
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[向量检索异常] {str(e)}")
            return []

    def keyword_search(self, query: str) -> List[Document]:
        """BM25关键词检索，异常捕获降级返回空列表"""
        if self.bm25_retriever is None:
            return []
        try:
            return self.bm25_retriever.invoke(query)
        except Exception as e:
            print(f"[BM25检索异常] {str(e)}")
            return []

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """文档去重，通过md5哈希"""
        seen = set()
        out = []
        for d in docs:
            h = calc_doc_hash(d)
            if h not in seen:
                seen.add(h)
                out.append(d)
        return out

    def hybrid_search(self, query: str, top_k: int = 4) -> List[Document]:
        """
        混合检索主入口：RRF倒数排名融合两路召回结果 → 去重 → Rerank重排序过滤噪声
        :param query: 用户查询问句（已经过LLM改写后的rewritten_query）
        :param top_k: 返回最终文档数量
        :return: 重排序后Document列表
        """
        # 1.两路召回，多拿候选给rerank做筛选
        vec_docs = self.vector_search(query, top_k=top_k * 2)
        key_docs = self.keyword_search(query)

        # 2.RRF融合打分
        scored_list = reciprocal_rank_fusion(vec_docs, key_docs)
        scored_list.sort(key=lambda x: x[0], reverse=True)
        fusion_docs = [item[1] for item in scored_list]

        # 3.去重
        fusion_docs = self._deduplicate_docs(fusion_docs)
        if len(fusion_docs) == 0:
            return []

        # 4.重排序（如果模型加载成功）
        if self.reranker is not None:
            score_input = [[query, doc.page_content] for doc in fusion_docs]
            rerank_scores = self.reranker.compute_score(score_input)
            scored_rerank = list(zip(rerank_scores, fusion_docs))
            # 过滤低分
            scored_rerank = [s for s in scored_rerank if s[0] >= self.rerank_threshold]
            scored_rerank.sort(key=lambda x: x[0], reverse=True)
            final_docs = [doc for score, doc in scored_rerank[:top_k]]
        else:
            # rerank不可用，降级直接取RRF结果
            final_docs = fusion_docs[:top_k]

        return final_docs


# ===================== 内嵌自测代码 =====================
if __name__ == "__main__":
    print("===== HybridRetriever RRF+Rerank混合检索模块自测 =====")
    # 模拟电商FAQ测试文档
    mock_docs = [
        Document(page_content="亚马逊订单退货：商品签收30天内支持无理由退货，运费买家承担", metadata={"category":"after_sale", "source":"faq.txt"}),
        Document(page_content="速卖通物流：普通邮政物流时效15‑25工作日，DHL快递5‑7天", metadata={"category":"logistics", "source":"faq.txt"}),
        Document(page_content="Listing标题规范：不能超过200字符，禁止堆砌关键词", metadata={"category":"listing_rule", "source":"faq.txt"}),
        Document(page_content="蓝牙耳机售后：非人为损坏享受12个月质保", metadata={"category":"after_sale", "source":"faq.txt"})
    ]

    retriever = HybridRetriever()
    retriever.build_bm25_index(mock_docs)
    # retriever.load_reranker()

    print("测试BM25关键词检索，查询：退货规则")
    bm25_res = retriever.keyword_search("退货规则")
    for idx, d in enumerate(bm25_res):
        print(f"[{idx+1}] {d.page_content}")

    print("\n测试hybrid_search（向量未初始化，仅BM25参与RRF）查询：退货规则")
    hybrid_res = retriever.hybrid_search("退货规则", top_k=3)
    for idx, d in enumerate(hybrid_res):
        print(f"[{idx+1}] {d.page_content}")

    print("\n✅ HybridRetriever RRF+Rerank版本自测完成")
    print("提示：真实业务需要手动调用 load_embedding()、load_reranker()")
    print("⚠️注意：BM25索引驻留内存，程序重启必须重新build_bm25_index")


def build_hybrid_retriever(llm):
    """
    工厂函数，供main.py/graph_builder导入调用
    注意：
    1. 此处没有加载embedding，没有构建BM25索引；
    2. build_bm25_index需要传入全部知识库切片文档；
    3. customer_service任务分支内部按需调用 load_embedding()、build_bm25_index()、load_reranker()
    """
    retriever = HybridRetriever()
    return retriever
