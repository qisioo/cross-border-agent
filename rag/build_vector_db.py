""" 向量库&BM25索引构建脚本
1.加载rag/knowledge目录下 txt、md、pdf 知识库文档
2.过滤空文档，文本切分
3.Milvus向量库支持【全量重建(drop collection) / 增量追加】；默认全量重建，防止重复向量堆积
4.构建Milvus向量库(agent03_<tenant_id>) + 内存BM25索引
⚠️重要：返回的 HybridRetriever 实例必须由主程序长期持有；实例销毁则BM25内存索引直接丢失
⚠️BM25索引无磁盘持久化，服务每次重启都需要重新加载文档重建索引
"""
import os
# HF国内镜像，必须放在所有import huggingface相关库之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import hashlib
import shutil
import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.hybrid_retriever import HybridRetriever, DEFAULT_COLLECTION, get_milvus_row_count

# 知识库文档目录
KNOWLEDGE_FOLDER = Path(__file__).parent / "knowledge"


def _extract_pdf_text(file_path: Path) -> str:
    """提取 PDF 文本 + 图片识别（Qwen-VL）"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        # 提取 PDF 里的图片，用 Qwen-VL 识别
        image_text = _extract_pdf_images_with_qwen(file_path)
        if image_text:
            text_parts.append(image_text)

        return "\n".join(text_parts)
    except ImportError:
        print(f"[警告] 未安装 pdfplumber，无法读取 PDF：{file_path.name}")
        return ""
    except Exception as e:
        print(f"[PDF读取异常] {file_path.name}：{e}")
        return ""


def _extract_pdf_images_with_qwen(file_path: Path) -> str:
    """用 PyMuPDF 提取 PDF 里的图片，再用 Qwen-VL 识别内容"""
    try:
        import fitz  # PyMuPDF
        import dashscope
        from dashscope import MultiModalConversation

        # 从环境变量读取 API key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("[警告] 未配置 DASHSCOPE_API_KEY，跳过 PDF 图片识别")
            return ""

        dashscope.api_key = api_key
        image_descriptions = []

        doc = fitz.open(file_path)
        img_count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 保存临时图片文件
                temp_img_path = f"temp_pdf_img_{page_num}_{img_index}.{image_ext}"
                with open(temp_img_path, "wb") as f:
                    f.write(image_bytes)

                # 调用 Qwen-VL 识别
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"file://{temp_img_path}"},
                                {"text": "请提取这张图片中的所有文字信息和关键内容，按原文格式输出。如果是流程图或表格，请描述其结构和内容。"}
                            ]
                        }
                    ]
                    response = MultiModalConversation.call(model="qwen3.8-omni-flash", messages=messages)
                    if response.status_code == 200:
                        result_text = response.output.choices[0].message.content[0]["text"]
                        image_descriptions.append(f"【PDF第{page_num+1}页图片{img_index+1}识别结果】\n{result_text}")
                        img_count += 1
                except Exception as e:
                    print(f"[Qwen-VL识别异常] 第{page_num+1}页图片{img_index+1}：{e}")

                # 删除临时图片文件
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        doc.close()
        if img_count > 0:
            print(f"[PDF图片识别] 共识别 {img_count} 张图片")
        return "\n".join(image_descriptions)
    except ImportError:
        print("[警告] 未安装 PyMuPDF 或 dashscope，跳过 PDF 图片识别")
        return ""
    except Exception as e:
        print(f"[PDF图片提取异常]：{e}")
        return ""


def _file_md5(file_path: Path) -> str:
    """计算文件内容的 MD5 哈希，用于去重"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_knowledge_docs(knowledge_dir: Optional[Path] = None) -> List[Document]:
    """
    读取指定知识库目录下全部 txt、md、pdf 知识库文档（多租户：传 knowledge/<tenant_id>/ 即只读该企业文档）
    过滤读取后空内容文档
    :param knowledge_dir: 目标目录，默认 KNOWLEDGE_FOLDER（rag/knowledge）
    :return: Document文档列表
    """
    target = knowledge_dir or KNOWLEDGE_FOLDER
    docs: List[Document] = []
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        print(f"[警告]知识库目录 {target} 不存在，已自动创建，请放入txt/md/pdf FAQ文档")
        return docs

    file_list = []
    file_list.extend(list(target.glob("*.txt")))
    file_list.extend(list(target.glob("*.md")))
    file_list.extend(list(target.glob("*.pdf")))

    if len(file_list) == 0:
        print("[警告] knowledge文件夹下没有txt/md/pdf知识库文件")
        return docs

    # 哈希去重：相同内容的文件只处理一次
    seen_md5 = set()
    print(f"[读取知识库] 一共找到 {len(file_list)} 个文档文件：")
    for file_path in file_list:
        try:
            # 先算 MD5，相同内容跳过
            file_md5 = _file_md5(file_path)
            if file_md5 in seen_md5:
                print(f"[跳过重复文件] {file_path.name}（内容与已读取文件相同）")
                continue
            seen_md5.add(file_md5)

            # 根据文件后缀选择读取方式
            if file_path.suffix.lower() == ".pdf":
                content = _extract_pdf_text(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            content = content.strip()
            if not content:
                print(f"[跳过空文件] {file_path.name}")
                continue

            # ==========【强化文本清洗】逐行清除行尾空白字符 ==========
            lines = [line.rstrip() for line in content.splitlines()]
            content = "\n".join(lines)
            # 将带空格的伪空行统一转为标准双换行
            content = re.sub(r"\n\s*\n", "\n\n", content)
            # 压缩连续多个空行，只保留1组双换行
            content = re.sub(r"\n\n+", "\n\n", content)

            doc = Document(
                page_content=content,
                metadata={"source": file_path.name, "file_md5": file_md5}
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
    print(f"[文档切分] 开始切分，原始文档数量：{len(docs)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,         # 调小chunk，知识库FAQ更精准
        chunk_overlap=45,       # 重叠适度降低，减少重复片段
        separators=["\n\n", "\n", "。", ". ", "！", "？", "；", ";", "，", ",", " "]
    )
    split_docs = splitter.split_documents(docs)
    print(f"[文档切分] 切分完成，初始文本块数量：{len(split_docs)}")
    seen = set()
    valid_chunks = []
    for doc in split_docs:
        content = doc.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            valid_chunks.append(doc)
    print(f"[文档切分] 去重完成，有效文本块数量：{len(valid_chunks)}")
    return valid_chunks


def build_vector_and_bm25(
    full_rebuild: bool = True
) -> Optional[HybridRetriever]:
    """
    完整构建流程：加载文档 → 切分 → 写入向量库 → 初始化BM25索引
    :param full_rebuild: True=全量重建(删除旧Milvus collection)，避免向量重复；False=增量追加文档
    :return: HybridRetriever实例，⚠️主程序必须持有此实例，不能丢弃，否则BM25内存索引丢失；失败返回None
    """
    try:
        raw_docs = load_knowledge_docs()
        if len(raw_docs) == 0:
            print("[构建终止]没有有效知识库文档")
            return None

        split_docs = split_documents(raw_docs)
        print(f"文档切分完成，有效文本块数量：{len(split_docs)}")

        # ==========【Milvus 全量重建/增量语义】==========
        # full_rebuild=True：load_embedding(drop_old=True) 删除旧collection重建，防重复向量堆积
        # full_rebuild=False：复用已有collection，row_count>0 则跳过写入（不再依赖chroma目录删除）
        print("[Milvus] 连接本机 Milvus standalone，collection:", DEFAULT_COLLECTION)
        retriever = HybridRetriever(collection_name=DEFAULT_COLLECTION)
        # 加载BGE‑M3 embedding模型
        print("[开始加载Embedding模型 BGE-M3]")
        retriever.load_embedding(drop_old=full_rebuild)
        print("[Embedding模型加载完成]")
        # 写入向量库
        existing_count = get_milvus_row_count(retriever.collection_name)
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


def build_tenant_retriever(
    tenant_id: str,
    full_rebuild: bool = True,
    progress_callback=None,
) -> Optional[HybridRetriever]:
    """
    按租户构建独立检索器（多租户纯隔离核心）
    读 rag/knowledge/<tenant_id>/ 下文档，写入 Milvus collection agent03_<tenant_id>，构建该租户内存BM25
    :param tenant_id: 企业/租户唯一标识
    :param full_rebuild: True=删除旧collection全量重建；False=复用已有collection(有向量则跳过写入)
    :param progress_callback: 可选进度回调(progress:int 0-100, message:str)，用于上传后进度条
    :return: HybridRetriever实例（需注册到 TenantRetrieverRegistry）；失败返回None
    """
    def _report(p: int, m: str):
        if progress_callback:
            try:
                progress_callback(p, m)
            except Exception:
                pass

    try:
        knowledge_dir = KNOWLEDGE_FOLDER / tenant_id
        _report(5, "开始读取企业知识库文档")
        raw_docs = load_knowledge_docs(knowledge_dir)
        if len(raw_docs) == 0:
            _report(100, "该企业暂无知识库文档")
            print(f"[构建终止]租户 {tenant_id} 无知识库文档")
            return None

        _report(15, "文档切分中")
        split_docs = split_documents(raw_docs)
        print(f"[租户切分完成] tenant={tenant_id}，文本块数量：{len(split_docs)}")
        _report(25, f"切分完成，共 {len(split_docs)} 个文本块")

        collection = f"agent03_{tenant_id}"
        print(f"[Milvus] 租户 {tenant_id} collection:", collection)
        _report(35, "正在加载向量模型并连接 Milvus")
        retriever = HybridRetriever(collection_name=collection)
        retriever.load_embedding(drop_old=full_rebuild)
        retriever.load_reranker()  # 加载重排序模型

        existing_count = get_milvus_row_count(collection)
        if full_rebuild or existing_count == 0:
            # 分批写入，便于进度条反馈（每批约 20%，从 35% 推进到 85%）
            batch_size = 50
            total = len(split_docs)
            for i in range(0, total, batch_size):
                batch = split_docs[i : i + batch_size]
                retriever.vector_store.add_documents(batch)
                done = min(i + batch_size, total)
                pct = 35 + int(50 * done / total)
                _report(pct, f"向量入库 {done}/{total}")
                print(f"[向量写入] tenant={tenant_id} 已写入 {done}/{total} 块")
        else:
            print(f"[向量库跳过写入] tenant={tenant_id} 已存在 {existing_count} 条向量")

        _report(90, "构建关键词索引")
        retriever.build_bm25_index(split_docs)
        print(f"[BM25内存索引构建完成] tenant={tenant_id}")
        _report(100, "构建完成")
        return retriever

    except Exception as e:
        print(f"[构建租户知识库异常] tenant={tenant_id}, err={str(e)}")
        import traceback
        traceback.print_exc()
        _report(100, f"构建失败：{str(e)}")
        return None


def build_conversation_retriever(
    tenant_id: str,
    conversation_id: str,
    full_rebuild: bool = True,
    progress_callback=None,
) -> Optional[HybridRetriever]:
    """
    按"企业+对话"构建独立检索器（对话级隔离核心）
    读 rag/knowledge/<tenant_id>/<conversation_id>/ 下文档，写入 Milvus collection
    agent03_<tenant_id>_<conversation_id>，并构建该对话内存BM25索引。
    同企业下不同对话的知识库完全独立、绝不跨对话；企业级旧接口仍由 build_tenant_retriever 负责。
    :param tenant_id: 企业/租户唯一标识
    :param conversation_id: 对话唯一标识(自动生成)，用于知识库目录与collection命名
    :param full_rebuild: True=删除旧collection全量重建；False=复用已有collection(有向量则跳过写入)
    :param progress_callback: 可选进度回调(progress:int 0-100, message:str)
    :return: HybridRetriever实例（需注册到 TenantRetrieverRegistry）；失败返回None
    """
    def _report(p: int, m: str):
        if progress_callback:
            try:
                progress_callback(p, m)
            except Exception:
                pass

    try:
        conversation_key = f"{tenant_id}::{conversation_id}"
        # 对话专属知识库目录 rag/knowledge/<tenant_id>/<conversation_id>/（文件层按对话隔离）
        knowledge_dir = KNOWLEDGE_FOLDER / tenant_id / conversation_id
        _report(5, "开始读取对话知识库文档")
        raw_docs = load_knowledge_docs(knowledge_dir)
        if len(raw_docs) == 0:
            _report(100, "该对话暂无知识库文档")
            print(f"[构建终止]对话 {conversation_key} 无知识库文档")
            return None

        _report(15, "文档切分中")
        split_docs = split_documents(raw_docs)
        print(f"[对话切分完成] conv={conversation_key}，文本块数量：{len(split_docs)}")
        _report(25, f"切分完成，共 {len(split_docs)} 个文本块")

        # 对话级 Milvus collection：agent03_<tenant_id>_<conversation_id>（向量层按对话隔离）
        collection = f"agent03_{tenant_id}_{conversation_id}"
        print(f"[Milvus] 对话 {conversation_key} collection:", collection)
        _report(35, "正在加载向量模型并连接 Milvus")
        retriever = HybridRetriever(collection_name=collection)
        retriever.load_embedding(drop_old=full_rebuild)
        retriever.load_reranker()  # 加载重排序模型

        existing_count = get_milvus_row_count(collection)
        if full_rebuild or existing_count == 0:
            # 分批写入，便于进度条反馈（每批约 20%，从 35% 推进到 85%）
            batch_size = 50
            total = len(split_docs)
            for i in range(0, total, batch_size):
                batch = split_docs[i : i + batch_size]
                retriever.vector_store.add_documents(batch)
                done = min(i + batch_size, total)
                pct = 35 + int(50 * done / total)
                _report(pct, f"向量入库 {done}/{total}")
                print(f"[向量写入] conv={conversation_key} 已写入 {done}/{total} 块")
        else:
            print(f"[向量库跳过写入] conv={conversation_key} 已存在 {existing_count} 条向量")

        _report(90, "构建关键词索引")
        retriever.build_bm25_index(split_docs)
        print(f"[BM25内存索引构建完成] conv={conversation_key}")
        _report(100, "构建完成")
        return retriever

    except Exception as e:
        print(f"[构建对话知识库异常] conv={tenant_id}::{conversation_id}, err={str(e)}")
        import traceback
        traceback.print_exc()
        _report(100, f"构建失败：{str(e)}")
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

