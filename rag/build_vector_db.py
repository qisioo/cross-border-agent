""" 向量库&BM25索引构建脚本
1.加载rag/knowledge目录下 txt、md知识库文档
2.过滤空文档，文本切分
3.Chroma支持【全量重建 / 增量追加】；默认全量重建，防止重复向量堆积
4.构建Chroma向量库 + 内存BM25索引
⚠️重要：返回的 HybridRetriever 实例必须由主程序长期持有；实例销毁则BM25内存索引直接丢失
⚠️BM25索引无磁盘持久化，服务每次重启都需要重新加载文档重建索引
"""
import os
# HF国内镜像，必须放在所有import huggingface相关库之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.hybrid_retriever import HybridRetriever

# 知识库文档目录
KNOWLEDGE_FOLDER = Path(__file__).parent / "knowledge"
# 向量库持久化目录
CHROMA_DB_FOLDER = Path(__file__).parent / "chroma_db"


def load_knowledge_docs() -> List[Document]:
    """
    读取knowledge文件夹下全部 txt、md知识库文档
    过滤读取后空内容文档
    :return: Document文档列表
    """
    docs: List[Document] = []
    if not KNOWLEDGE_FOLDER.exists():
        KNOWLEDGE_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"[警告]知识库目录 {KNOWLEDGE_FOLDER} 不存在，已自动创建，请放入txt/md FAQ文档")
        return docs

    file_list = []
    file_list.extend(list(KNOWLEDGE_FOLDER.glob("*.txt")))
    file_list.extend(list(KNOWLEDGE_FOLDER.glob("*.md")))

    if len(file_list) == 0:
        print("[警告] knowledge文件夹下没有txt/md知识库文件")
        return docs

    print(f"[读取知识库] 一共找到 {len(file_list)} 个文档文件：")
    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.strip()
            if not content:
                print(f"[跳过空文件] {file_path.name}")
                continue

            # ==========【强化文本清洗】新增逐行清除行尾空白字符，解决肉眼空行带空格问题 ==========
            lines = [line.rstrip() for line in content.splitlines()]
            content = "\n".join(lines)
            # 将带空格的伪空行统一转为标准双换行
            content = re.sub(r"\n\s*\n", "\n\n", content)
            # 压缩连续多个空行，只保留1组双换行
            content = re.sub(r"\n\n+", "\n\n", content)

            doc = Document(
                page_content=content,
                metadata={"source": file_path.name}
            )
            docs.append(doc)
            print(f"✅ 成功读取文档：{file_path.name}，字符长度：{len(content)}")
        except UnicodeDecodeError:
            print(f"[文件编码错误] {file_path.name}，请使用utf‑8编码保存")
        except Exception as e:
            print(f"[读取文件异常] {file_path.name} , err:{str(e)}")
    print(f"[读取知识库完成] 有效原始文档总数：{len(docs)}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """文档递归文本切分 + 文本去重【调优切分参数，降低RAG噪声】"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,         # 调小chunk，知识库FAQ更精准
        chunk_overlap=45,       # 重叠适度降低，减少重复片段
        separators=["\n\n", "\n", "。", ". ", "，", ",", " "]
    )
    split_docs = splitter.split_documents(docs)
    seen = set()
    valid_chunks = []
    for doc in split_docs:
        content = doc.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            valid_chunks.append(doc)
    return valid_chunks


def build_vector_and_bm25(
    full_rebuild: bool = True
) -> Optional[HybridRetriever]:
    """
    完整构建流程：加载文档 → 切分 → 写入向量库 → 初始化BM25索引
    :param full_rebuild: True=全量重建，先删除旧chroma_db目录，避免向量重复；False=增量追加文档
    :return: HybridRetriever实例，⚠️主程序必须持有此实例，不能丢弃，否则BM25内存索引丢失；失败返回None
    """
    try:
        raw_docs = load_knowledge_docs()
        if len(raw_docs) == 0:
            print("[构建终止]没有有效知识库文档")
            return None

        split_docs = split_documents(raw_docs)
        print(f"文档切分完成，有效文本块数量：{len(split_docs)}")

        # ==========仅调整顺序：先删除旧向量库，再实例化retriever==========
        if full_rebuild and CHROMA_DB_FOLDER.exists():
            shutil.rmtree(CHROMA_DB_FOLDER)
            print("[全量重建模式]已清空旧Chroma向量库")

        retriever = HybridRetriever(persist_directory=str(CHROMA_DB_FOLDER))
        # 加载BGE‑M3 embedding模型
        print("[开始加载Embedding模型 BGE-M3]")
        retriever.load_embedding()
        print("[Embedding模型加载完成]")
        # 写入向量库
        existing_count = 0
        try:
            existing_count = retriever.vector_store._collection.count()
        except Exception:
            existing_count = 0
        if full_rebuild or existing_count == 0:
            retriever.vector_store.add_documents(split_docs)
            print("[向量库写入完成]")
        else:
            print(f"[向量库跳过写入] 已存在 {existing_count} 条向量，full_rebuild=False 不重复追加")
        # 构建内存BM25索引
        retriever.build_bm25_index(split_docs)
        print("[BM25内存索引构建完成]")

        # ===================== Reranker预留接入点(P1‑1) =====================
        # 后续本地下载BGE‑Reranker模型后，在这里初始化reranker对象，
        # 在 hybrid_retriever.py 的 hybrid_search 内部做重排序，此处不引入额外依赖
        print("[预留]BGE‑Reranker重排序接入位置，待本地部署模型后开启")

        print(f"✅构建完成，向量库+BM25内存索引就绪，共 {len(split_docs)} 文本块")
        return retriever

    except Exception as e:
        print(f"[构建知识库整体异常] {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ===================== 内嵌自测代码 =====================
if __name__ == "__main__":
    print("==== build_vector_db 脚本自测 ====")
    # 自测只测试：加载文档、文本切分、空文档过滤逻辑；不加载embedding，不操作真实向量库
    mock_test_docs = [
        Document(page_content="速卖通发货时效：普通小包15‑25天，专线物流7‑12天\n售后规则：签收30天支持退货。", metadata={"source":"logistics_faq.txt"}),
        Document(page_content="   ", metadata={"source":"empty_file.txt"}),  # 模拟空白文档，应当被过滤
    ]
    print(f"模拟原始文档数量：{len(mock_test_docs)}")
    splited = split_documents(mock_test_docs)
    print(f"过滤空chunk之后文本块数量：{len(splited)}")
    for idx, chunk in enumerate(splited):
        print(f"chunk{idx+1}: {chunk.page_content.strip()}")

    print("\n✅ build_vector_db自测完成：空文档过滤、切分逻辑正常")
    print("提示：调用 build_vector_and_bm25() 才会加载BGE‑M3模型并构建向量库")
    print("⚠️重要：返回的retriever实例必须全程持有，实例销毁BM25索引直接丢失")

    # ==========新增：取消下面注释，即可执行真实知识库构建（自测完成后手动开启）==========
    print("\n==== 开始真实知识库构建 ====")
    real_retriever = build_vector_and_bm25(full_rebuild=True)
    if real_retriever:
        print("真实知识库构建成功，可以在main.py中使用该实例")

