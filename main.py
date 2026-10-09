""" FastAPI 主入口 优化点：异步ainvoke、输入校验、超时控制、返回字段过滤、异常捕获 """
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"


import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph_builder import build_graph
from rag.build_vector_db import build_vector_and_bm25
from utils.helper import generate_trace_id
# 全局agent graph实例
agent_graph = None

from rag.build_vector_db import build_vector_and_bm25

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph
    print("===== Agent Graph 初始化 =====")
    # ==========【Skill规则预加载】启动时预加载skill/rules全部分片进内存（渐进式加载Level1），Skill结果优先 ==========
    from utils.skill_loader import get_skill_loader
    get_skill_loader()
    # ======================================================================
    # 构建真实向量库和BM25检索器
    hybrid_retriever = build_vector_and_bm25(full_rebuild=False)
    hybrid_retriever.load_reranker()
    agent_graph = build_graph(hybrid_retriever)
    yield
    print("===== 应用退出 =====")


app = FastAPI(title="Agent03智能体服务", lifespan=lifespan)

# 请求体模型
class AgentRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None

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
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # ==========【读取历史对话消息，丢弃历史任务状态 ==========
    try:
        saved_state = await agent_graph.get_state(config)

        history_messages = saved_state.values.get("messages", [])
    except Exception:
        history_messages = []
    new_messages = history_messages + [HumanMessage(content=user_query)]

    payload = {
        "messages": new_messages
    }
    # ==========【新增核心修复：每次请求前置重置checkpoint里的agent_state】 ==========
    from agent.state import AgentState
    clean_agent_state = AgentState(
        user_query=user_query,
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


@app.get("/health")
async def health_check():
    if agent_graph is None:
        return {"status": "unavailable", "reason": "agent graph not initialized"}
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
