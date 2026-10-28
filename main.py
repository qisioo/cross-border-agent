""" FastAPI 主入口 优化点：异步ainvoke、输入校验、超时控制、返回字段过滤、异常捕获 """
import os
from dotenv import load_dotenv
# 显式加载 .env 配置（DEEPSEEK_API_KEY 等）；override=True 覆盖进程残留旧值，保证用 .env 的最新密钥
load_dotenv(override=True)
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"


import time
import asyncio
import json
import uuid
import threading
from typing import Optional, Dict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph_builder import build_graph
from rag.tenant_registry import get_tenant_registry
# ==========【对话级隔离】企业主ID下多对话管理：对话元数据增删改查（conversations.json） ==========
from rag.conversation_store import (
    list_conversations,
    create_conversation,
    rename_conversation,
    delete_conversation,
    get_conversation,
)
from utils.helper import generate_trace_id
# 全局agent graph实例
agent_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph
    print("===== Agent Graph 初始化 =====")
    # ==========【Skill规则预加载】启动时预加载skill/rules全部分片进内存（渐进式加载Level1），Skill结果优先 ==========
    from utils.skill_loader import get_skill_loader
    get_skill_loader()
    # ======================================================================
    # 多租户纯隔离：启动创建租户检索注册表（不构建公共默认库）
    # 每个企业的知识库由上传接口按 tenant_id 单独构建并注册进来
    tenant_registry = get_tenant_registry()
    agent_graph = build_graph(tenant_registry)
    print("===== Agent Graph 初始化完成（多租户纯隔离模式） =====")
    yield
    print("===== 应用退出 =====")


app = FastAPI(title="Agent03智能体服务", lifespan=lifespan)

# ==========【多租户：企业知识库上传 + 异步构建 + 进度查询】==========
# BUILD_TASKS: task_id -> {"task_id","tenant_id","status","progress"(0-100),"message"}
# 构建在后台线程执行，前端轮询进度接口展示"正在构建向量数据库 + 进度条"
BUILD_TASKS: Dict[str, dict] = {}


@app.post("/api/tenant/upload")
async def upload_tenant_knowledge(tenant_id: str = Form(...), file: UploadFile = File(...)):
    """上传企业知识库文档，异步构建该企业独立向量库(agent03_<tenant_id>)与BM25索引，返回task_id供进度查询"""
    import re
    tid = (tenant_id or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail="tenant_id不能为空")
    # 校验租户id：只允许字母/数字/下划线/连字符（用于Milvus collection命名与目录），长度1-64
    if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", tid):
        raise HTTPException(status_code=400, detail="tenant_id只能包含字母/数字/下划线/连字符，长度1-64")

    filename = file.filename or "knowledge.txt"
    if not (filename.lower().endswith((".txt", ".md", ".csv", ".pdf"))):
        raise HTTPException(status_code=400, detail="仅支持 txt/md/csv/pdf 文档")

    # 保存到该企业专属知识库目录 rag/knowledge/<tenant_id>/（文件层隔离）
    tenant_dir = Path("rag/knowledge") / tid
    tenant_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name  # 防路径穿越
    save_path = tenant_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)
    print(f"[upload_tenant_knowledge] 已保存租户 {tid} 文档: {save_path}")

    task_id = uuid.uuid4().hex
    BUILD_TASKS[task_id] = {
        "task_id": task_id, "tenant_id": tid, "status": "running",
        "progress": 0, "message": "正在构建向量数据库...",
    }
    # 后台线程异步构建，不阻塞上传请求
    t = threading.Thread(target=_run_tenant_build, args=(task_id, tid), daemon=True)
    t.start()
    return {"task_id": task_id, "tenant_id": tid, "status": "running", "message": "正在构建向量数据库..."}


def _run_tenant_build(task_id: str, tenant_id: str):
    """后台构建租户知识库并注册到租户注册表；进度写入BUILD_TASKS供进度接口轮询"""
    from rag.build_vector_db import build_tenant_retriever
    from rag.tenant_registry import get_tenant_registry

    def _cb(p, m):
        BUILD_TASKS[task_id]["progress"] = p
        BUILD_TASKS[task_id]["message"] = m

    BUILD_TASKS[task_id]["status"] = "running"
    retriever = build_tenant_retriever(tenant_id, full_rebuild=True, progress_callback=_cb)
    if retriever is None:
        BUILD_TASKS[task_id].update({"status": "failed", "progress": 100, "message": "构建失败，请检查文档内容"})
    else:
        get_tenant_registry().register(tenant_id, retriever)
        BUILD_TASKS[task_id].update({"status": "done", "progress": 100, "message": "构建完成"})


@app.get("/api/tenant/build/{task_id}")
async def get_build_progress(task_id: str):
    """查询企业知识库构建进度，前端用于进度条轮询"""
    task = BUILD_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="构建任务不存在")
    return task

# ==========【对话级隔离：企业主ID下的多对话管理 + 对话内知识库增删改查】==========
# 结构：企业(tenant_id) → 多个对话(conversation) → 每对话独立知识库(文件/向量/检索三层隔离，绝不跨对话)
def _validate_tenant(tenant_id: str) -> str:
    """校验企业id：只允许字母/数字/下划线/连字符，长度1-64（用于Milvus collection与目录命名）"""
    import re
    tid = (tenant_id or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail="tenant_id不能为空")
    if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", tid):
        raise HTTPException(status_code=400, detail="tenant_id只能包含字母/数字/下划线/连字符，长度1-64")
    return tid


def _validate_conversation_id(conversation_id: str) -> str:
    """校验对话id：只允许字母/数字（uuid hex），长度1-64"""
    import re
    cid = (conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id不能为空")
    if not re.match(r"^[A-Za-z0-9]{1,64}$", cid):
        raise HTTPException(status_code=400, detail="conversation_id只能包含字母/数字，长度1-64")
    return cid


class ConversationCreate(BaseModel):
    name: Optional[str] = None


class ConversationRename(BaseModel):
    name: str = ""


@app.post("/api/tenant/{tenant_id}/conversations")
async def create_conversation_api(tenant_id: str, body: ConversationCreate):
    """创建对话：企业主ID下新增一个对话，id自动生成，name可指定(默认"对话N")，并创建对话专属知识库目录"""
    tid = _validate_tenant(tenant_id)
    conv = create_conversation(tid, body.name or "")
    print(f"[create_conversation] 企业 {tid} 新建对话: {conv}")
    return {"code": 0, "data": conv}


@app.get("/api/tenant/{tenant_id}/conversations")
async def list_conversations_api(tenant_id: str):
    """列出某企业全部对话（不跨对话）"""
    tid = _validate_tenant(tenant_id)
    return {"code": 0, "data": list_conversations(tid)}


@app.put("/api/tenant/{tenant_id}/conversations/{conversation_id}")
async def rename_conversation_api(tenant_id: str, conversation_id: str, body: ConversationRename):
    """改对话显示名（id不变，知识库不变）"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    conv = rename_conversation(tid, cid, body.name or "")
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"code": 0, "data": conv}


@app.delete("/api/tenant/{tenant_id}/conversations/{conversation_id}")
async def delete_conversation_api(tenant_id: str, conversation_id: str):
    """删除对话：清对话元数据 + 知识库文件目录 + Milvus向量库 + 注册表，彻底隔离清理"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    ok = delete_conversation(tid, cid)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    # 清理对话知识库文件目录
    import shutil
    conv_dir = Path("rag/knowledge") / tid / cid
    if conv_dir.exists():
        shutil.rmtree(conv_dir, ignore_errors=True)
    # 清理注册表与Milvus向量库
    registry = get_tenant_registry()
    registry.remove(f"{tid}::{cid}")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://127.0.0.1:19530")
        coll = f"agent03_{tid}_{cid}"
        if client.has_collection(coll):
            client.drop_collection(coll)
    except Exception as e:
        print(f"[delete_conversation] Milvus清理异常(可忽略): {e}")
    print(f"[delete_conversation] 已删除企业 {tid} 对话 {cid}")
    return {"code": 0, "msg": "对话已删除", "conversation_id": cid}


@app.get("/api/tenant/{tenant_id}/conversations/{conversation_id}")
async def conversation_detail_api(tenant_id: str, conversation_id: str):
    """对话详情：元数据 + 知识库文件列表 + 构建状态"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    conv = get_conversation(tid, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    conv_dir = Path("rag/knowledge") / tid / cid
    files = []
    if conv_dir.exists():
        for fp in sorted(conv_dir.glob("*")):
            if fp.is_file():
                files.append({"name": fp.name, "size": fp.stat().st_size})
    from rag.hybrid_retriever import get_milvus_row_count
    retriever = get_tenant_registry().get_for(tid, cid)
    built = retriever is not None
    row_count = 0
    if built:
        try:
            row_count = get_milvus_row_count(f"agent03_{tid}_{cid}")
        except Exception:
            row_count = 0
    data = dict(conv)
    data["knowledge"] = files
    data["built"] = built
    data["chunk_count"] = row_count
    return {"code": 0, "data": data}


# 对话级知识库构建（复用 BUILD_TASKS 进度机制，加 conversation_id 标识）
def _run_conversation_build(task_id: str, tenant_id: str, conversation_id: str):
    """后台构建对话知识库并注册到注册表(键 tenant::conv)；进度写入BUILD_TASKS供进度接口轮询。
    若对话下已无任何文档(用户删光了知识库文件)，则清理旧collection与注册状态，避免向量库残留。
    """
    from rag.build_vector_db import build_conversation_retriever

    def _cb(p, m):
        BUILD_TASKS[task_id]["progress"] = p
        BUILD_TASKS[task_id]["message"] = m

    BUILD_TASKS[task_id]["status"] = "running"
    retriever = build_conversation_retriever(tenant_id, conversation_id, full_rebuild=True, progress_callback=_cb)
    if retriever is None:
        msg = BUILD_TASKS[task_id].get("message", "")
        if "无知识库文档" in msg:
            # 对话下已没有任何文档：移除旧注册项并drop旧collection，避免向量残留与built状态失真
            get_tenant_registry().remove(f"{tenant_id}::{conversation_id}")
            try:
                from pymilvus import MilvusClient
                client = MilvusClient(uri="http://127.0.0.1:19530")
                coll = f"agent03_{tenant_id}_{conversation_id}"
                if client.has_collection(coll):
                    client.drop_collection(coll)
                    print(f"[_run_conversation_build] 对话 {tenant_id}::{conversation_id} 无文档，已drop旧collection {coll}")
            except Exception as e:
                print(f"[_run_conversation_build] 清理collection异常(可忽略): {e}")
            BUILD_TASKS[task_id].update({"status": "done", "progress": 100, "message": "无知识库文档，已清空该对话向量库"})
        else:
            BUILD_TASKS[task_id].update({"status": "failed", "progress": 100, "message": "构建失败，请检查文档内容"})
    else:
        get_tenant_registry().register(f"{tenant_id}::{conversation_id}", retriever)
        BUILD_TASKS[task_id].update({"status": "done", "progress": 100, "message": "构建完成"})


@app.post("/api/tenant/{tenant_id}/conversations/{conversation_id}/upload")
async def upload_conversation_knowledge(tenant_id: str, conversation_id: str, file: UploadFile = File(...)):
    """上传对话知识库文档，异步构建该对话独立向量库(agent03_<tenant>_<conv>)，返回task_id供进度查询"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    if get_conversation(tid, cid) is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    filename = file.filename or "knowledge.txt"
    if not (filename.lower().endswith((".txt", ".md", ".csv", ".pdf"))):
        raise HTTPException(status_code=400, detail="仅支持 txt/md/csv/pdf 文档")

    # 保存到该对话专属知识库目录 rag/knowledge/<tenant>/<conversation>/（文件层按对话隔离）
    conv_dir = Path("rag/knowledge") / tid / cid
    conv_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name  # 防路径穿越
    save_path = conv_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)
    print(f"[upload_conversation_knowledge] 已保存对话 {tid}::{cid} 文档: {save_path}")

    task_id = uuid.uuid4().hex
    BUILD_TASKS[task_id] = {
        "task_id": task_id, "tenant_id": tid, "conversation_id": cid,
        "status": "running", "progress": 0, "message": "正在构建向量数据库...",
    }
    t = threading.Thread(target=_run_conversation_build, args=(task_id, tid, cid), daemon=True)
    t.start()
    return {"task_id": task_id, "tenant_id": tid, "conversation_id": cid, "status": "running", "message": "正在构建向量数据库..."}


@app.get("/api/tenant/{tenant_id}/conversations/{conversation_id}/knowledge")
async def list_conversation_knowledge(tenant_id: str, conversation_id: str):
    """列出某对话知识库文件 + 构建状态（chunk数）"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    if get_conversation(tid, cid) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    conv_dir = Path("rag/knowledge") / tid / cid
    files = []
    if conv_dir.exists():
        for fp in sorted(conv_dir.glob("*")):
            if fp.is_file():
                files.append({"name": fp.name, "size": fp.stat().st_size})
    from rag.hybrid_retriever import get_milvus_row_count
    retriever = get_tenant_registry().get_for(tid, cid)
    built = retriever is not None
    row_count = 0
    if built:
        try:
            row_count = get_milvus_row_count(f"agent03_{tid}_{cid}")
        except Exception:
            row_count = 0
    return {"code": 0, "data": {"conversation_id": cid, "files": files, "built": built, "chunk_count": row_count}}


@app.delete("/api/tenant/{tenant_id}/conversations/{conversation_id}/knowledge/{filename}")
async def delete_conversation_knowledge_file(tenant_id: str, conversation_id: str, filename: str):
    """删除对话下某个知识库文件，并触发该对话知识库重建"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    if get_conversation(tid, cid) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    safe_name = Path(filename).name  # 防路径穿越
    file_path = Path("rag/knowledge") / tid / cid / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    file_path.unlink()
    print(f"[delete_conversation_knowledge_file] 已删除对话 {tid}::{cid} 文件: {safe_name}，触发重建")
    task_id = uuid.uuid4().hex
    BUILD_TASKS[task_id] = {
        "task_id": task_id, "tenant_id": tid, "conversation_id": cid,
        "status": "running", "progress": 0, "message": "正在重建向量数据库...",
    }
    t = threading.Thread(target=_run_conversation_build, args=(task_id, tid, cid), daemon=True)
    t.start()
    return {"task_id": task_id, "conversation_id": cid, "status": "running", "message": "已删除文件，正在重建知识库"}


@app.post("/api/tenant/{tenant_id}/conversations/{conversation_id}/rebuild")
async def rebuild_conversation_knowledge(tenant_id: str, conversation_id: str):
    """手动重建某对话知识库（重新索引现有全部文件）"""
    tid = _validate_tenant(tenant_id)
    cid = _validate_conversation_id(conversation_id)
    if get_conversation(tid, cid) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    task_id = uuid.uuid4().hex
    BUILD_TASKS[task_id] = {
        "task_id": task_id, "tenant_id": tid, "conversation_id": cid,
        "status": "running", "progress": 0, "message": "正在重建向量数据库...",
    }
    t = threading.Thread(target=_run_conversation_build, args=(task_id, tid, cid), daemon=True)
    t.start()
    return {"task_id": task_id, "conversation_id": cid, "status": "running", "message": "正在重建向量数据库"}

# 请求体模型
class AgentRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None
    tenant_id: str = "default"
    conversation_id: Optional[str] = None

# 配置常量
MAX_QUERY_LENGTH = 800
AGENT_RUN_TIMEOUT = 60
FILTER_OUT_FIELDS = {
    "trace_id",
    "retry_count",
    "current_task_index",
    "task_list",
    "error_msg",
    "original_input"
}

def filter_agent_output(full_state: dict) -> dict:
    return {k: v for k, v in full_state.items() if k not in FILTER_OUT_FIELDS}


@app.post("/api/agent/run")
async def agent_run(req: AgentRequest):
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent服务尚未初始化，请稍后重试")

    user_query = req.user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="user_query不能为空")
    if len(user_query) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"输入不能超过{MAX_QUERY_LENGTH}字符")

    trace_id = generate_trace_id()
    print(f"[api][trace_id={trace_id}] receive request, query={user_query[:120]}")

    session_id = req.session_id or "sess-default-001"
    tenant_id = (req.tenant_id or "default").strip() or "default"
    conversation_id = (req.conversation_id or "").strip() or ""
    # 多租户 + 对话级隔离：thread_id 前缀企业id与对话id，隔离不同企业/不同对话的历史与checkpoint，绝不跨对话
    thread_id = f"{tenant_id}:{conversation_id}:{session_id}" if conversation_id else f"{tenant_id}:{session_id}"
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ==========【LangGraph checkpoint 自动恢复历史 messages，只需传新消息】 ==========
    # 注意：不要手动拼 history_messages + 新消息，因为 add_messages reducer 会把
    # 传入的无 id 历史消息全部当作新消息重复追加，导致历史错乱。
    payload = {
        "messages": [HumanMessage(content=user_query)]
    }
    # ==========【新增核心修复：每次请求前置重置checkpoint里的agent_state】 ==========
    from agent.state import AgentState
    clean_agent_state = AgentState(
        user_query=user_query,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        original_input=user_query,
        product_raw_info=None,
        competitor_report="",
        listing_content=None,
        customer_qa_result=None,
        sales_data=None,
        current_task_type="",
        task_list=[],
        current_task_index=0,
        retry_count=0,
        error_msg="",
        hit_boundary=False,
        boundary_answer="",
        answer_source="",
        final_answer="",
        trace_id=trace_id
    )
    # 覆盖checkpoint中残留的agent_state任务数据
    agent_graph.update_state(config, {"agent_state": clean_agent_state})

    # =================================================================================

    try:
        start_ts = time.time()
        result = await asyncio.wait_for(
            agent_graph.ainvoke(payload, config=config),
            timeout=AGENT_RUN_TIMEOUT
        )
        cost = round(time.time() - start_ts, 3)

        agent_state_full = result["agent_state"].model_dump()
        print(f"[api][trace_id={trace_id}] agent执行完成，耗时={cost}s")

        final_answer = ""
        agent_state_obj = result.get("agent_state")
        if agent_state_obj and hasattr(agent_state_obj, "final_answer") and agent_state_obj.final_answer:
            final_answer = agent_state_obj.final_answer
        elif agent_state_obj and hasattr(agent_state_obj, "competitor_report") and agent_state_obj.competitor_report:
            final_answer = agent_state_obj.competitor_report
        else:
            msg_list = result.get("messages", [])
            if msg_list and isinstance(msg_list[-1], AIMessage):
                final_answer = msg_list[-1].content

        # ==========【兜底正则清洗，干掉子任务标记】 ==========
        import re
        pattern = re.compile(r"---【子任务\d+】.*?\n|【子任务中间结果】", re.MULTILINE)
        final_answer = pattern.sub("", final_answer)
        final_answer = re.sub(r"\n\s*\n", "\n\n", final_answer).strip()
        # =====================================================

        print(f"[api][trace_id={trace_id}] 读取final_answer字段：{final_answer[:200]}")

        safe_output = filter_agent_output(agent_state_full)
        return {
            "code": 0,
            "msg": "success",
            "trace_id": trace_id,
            "cost_seconds": cost,
            "final_answer": final_answer,
            "data": safe_output
        }

    except asyncio.TimeoutError:
        print(f"[api][trace_id={trace_id}] agent执行超时 {AGENT_RUN_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Agent执行超时({AGENT_RUN_TIMEOUT}秒)，竞品分析任务耗时较长，请简化查询条件")
    except Exception as e:
        err_msg = str(e)
        print(f"[api][trace_id={trace_id}] 接口异常：{err_msg}")
        raise HTTPException(status_code=500, detail=f"服务内部异常：{err_msg}")


# ==========【SSE 流式接口（LangGraph 原生流式）】==========
# 节点名称到中文提示的映射
NODE_NAME_MAP = {
    "pre_rag_retrieve": "正在前置检索...",
    "intent_split": "正在切分意图...",
    "reflection_check": "正在反思校验...",
    "intent_execute": "正在执行任务...",
    "result_collect": "正在汇总结果...",
    "output_reflection": "正在输出校验...",
}


@app.post("/api/agent/stream")
async def agent_stream(req: AgentRequest):
    """SSE 流式接口（LangGraph 原生流式）：推送节点执行过程 + token 级流式输出"""
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent服务尚未初始化，请稍后重试")
    user_query = req.user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="user_query不能为空")
    if len(user_query) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"输入不能超过{MAX_QUERY_LENGTH}字符")

    trace_id = generate_trace_id()
    print(f"[api/stream][trace_id={trace_id}] receive request, query={user_query[:120]}")

    session_id = req.session_id or "sess-default-001"
    tenant_id = (req.tenant_id or "default").strip() or "default"
    conversation_id = (req.conversation_id or "").strip() or ""
    thread_id = f"{tenant_id}:{conversation_id}:{session_id}" if conversation_id else f"{tenant_id}:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 前置重置 checkpoint 里的 agent_state
    from agent.state import AgentState
    clean_agent_state = AgentState(
        user_query=user_query,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        original_input=user_query,
        product_raw_info=None,
        competitor_report="",
        listing_content=None,
        customer_qa_result=None,
        sales_data=None,
        current_task_type="",
        task_list=[],
        current_task_index=0,
        retry_count=0,
        error_msg="",
        hit_boundary=False,
        boundary_answer="",
        answer_source="",
        final_answer="",
        trace_id=trace_id
    )
    agent_graph.update_state(config, {"agent_state": clean_agent_state})

    payload = {"messages": [HumanMessage(content=user_query)]}

    async def event_gen():
        try:
            # 起始事件
            yield f"data: {json.dumps({'type': 'start', 'trace_id': trace_id}, ensure_ascii=False)}\n\n"

            final_answer = ""

            # 用 LangGraph 的原生流式能力，stream_mode="updates" 推送节点执行过程
            async for event in agent_graph.astream(payload, config=config, stream_mode="updates"):
                # event 是一个字典，key 是节点名称，value 是该节点的输出
                for node_name, node_output in event.items():
                    # 推送节点执行提示
                    node_hint = NODE_NAME_MAP.get(node_name, f"正在执行 {node_name}...")
                    yield f"data: {json.dumps({'type': 'node', 'node': node_name, 'hint': node_hint}, ensure_ascii=False)}\n\n"

                    # 检查是否有 final_answer 输出
                    if isinstance(node_output, dict) and "agent_state" in node_output:
                        agent_state = node_output["agent_state"]
                        if hasattr(agent_state, "final_answer") and agent_state.final_answer:
                            final_answer = agent_state.final_answer

            # 兜底：如果没有从节点输出拿到 final_answer，从结果里拿
            if not final_answer:
                result = await asyncio.wait_for(
                    agent_graph.ainvoke(payload, config=config),
                    timeout=AGENT_RUN_TIMEOUT
                )
                agent_state_obj = result.get("agent_state")
                if agent_state_obj and hasattr(agent_state_obj, "final_answer") and agent_state_obj.final_answer:
                    final_answer = agent_state_obj.final_answer
                elif agent_state_obj and hasattr(agent_state_obj, "competitor_report") and agent_state_obj.competitor_report:
                    final_answer = agent_state_obj.competitor_report
                else:
                    msg_list = result.get("messages", [])
                    if msg_list and isinstance(msg_list[-1], AIMessage):
                        final_answer = msg_list[-1].content

            # 正则清洗
            import re
            pattern = re.compile(r"---【子任务\d+】.*?\n|【子任务中间结果】", re.MULTILINE)
            final_answer = pattern.sub("", final_answer)
            final_answer = re.sub(r"\n\s*\n", "\n\n", final_answer).strip()

            print(f"[api/stream][trace_id={trace_id}] 执行完成，回答长度={len(final_answer)}")

            # 推送 token 级流式输出（模拟打字机效果）
            step = 4
            for i in range(0, len(final_answer), step):
                chunk = final_answer[i:i + step]
                yield f"data: {json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)

            # 结束事件
            yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id}, ensure_ascii=False)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent执行超时({AGENT_RUN_TIMEOUT}秒)'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[api/stream][trace_id={trace_id}] 异常: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'服务内部异常：{str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    if agent_graph is None:
        return {"status": "unavailable", "reason": "agent graph not initialized"}
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
