""" LangGraph 构图模块 graph_builder.py 组装全部节点、条件边，返回可编译运行的Agent图实例
外部 main.py 需要传入全局的 hybrid_retriever (RAG混合检索对象)
改造后工作流：
pre_rag_retrieve_node → query_rewrite_node → boundary_judge_node → need_knowledge_judge_node
    ├ need_retrieve=True → do_retrieve_node → task_parser_node → task_router → exec_task_dispatcher → task_router（循环执行子任务）
    └ need_retrieve=False → task_parser_node → task_router → exec_task_dispatcher → task_router（循环执行子任务）
全部任务完成 / 报错 / 重试超限 / hit_boundary=True → result_collect_node → END
"""
from typing import Any
from functools import partial
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from agent.state import GraphState
from langgraph.checkpoint.memory import MemorySaver
from utils.helper import update_agent_state, generate_trace_id

# ==========【更新导入：新增query_rewrite_node】==========
from agent.nodes import (
    pre_rag_retrieve_node,
    query_rewrite_node,
    boundary_judge_node,
    route_after_boundary,
    need_knowledge_judge_node,
    route_knowledge_judge,
    do_retrieve_node,
    task_parser_node,
    task_router,
    exec_task_dispatcher,
    result_collect_node,
    increase_task_index
)

from agent.tools import (
    product_info_fetch,
    competitor_analysis,
    listing_generate,
    sales_data_query,
    customer_service_qa
)

# ===================== 已删除此处旧版exec_task_dispatcher函数，统一使用agent.nodes.exec_task_dispatcher =====================
def build_graph(hybrid_retriever):
    """
    构建并返回编译后的 StateGraph 实例
    :param hybrid_retriever: 全局混合检索对象 main.py从rag.build_vector_db获取传入
    :return: compiled_graph
    """
    memory_saver = MemorySaver()
    graph = StateGraph(GraphState)

    # ========= 注册节点【新增 query_rewrite】 =========
    graph.add_node("pre_rag_retrieve", pre_rag_retrieve_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("boundary_judge", boundary_judge_node)
    graph.add_node("knowledge_judge", need_knowledge_judge_node)
    graph.add_node("do_retrieve", partial(do_retrieve_node, hybrid_retriever=hybrid_retriever))
    graph.add_node("task_parser", task_parser_node)
    # 使用 partial 注入参数，规避lambda闭包捕获问题，调用nodes内的exec_task_dispatcher
    graph.add_node("exec_task", partial(exec_task_dispatcher, hybrid_retriever=hybrid_retriever))
    graph.add_node("result_collect", result_collect_node)

    # ========= 设置入口点【启动首先运行 pre_rag_retrieve】 =========
    graph.set_entry_point("pre_rag_retrieve")

    # ========= 修改链路：pre_rag_retrieve → query_rewrite → boundary_judge =========
    graph.add_edge("pre_rag_retrieve", "query_rewrite")
    graph.add_edge("query_rewrite", "boundary_judge")

    # ========= 边界条件路由 =========
    graph.add_conditional_edges(
        source="boundary_judge",
        path=route_after_boundary,
        path_map={
            "task_parser_node": "knowledge_judge",
            "result_collect": "result_collect"
        }
    )

    # ========= 新增：知识相关性判断路由 =========
    graph.add_conditional_edges(
        source="knowledge_judge",
        path=route_knowledge_judge,
        path_map={
            "do_retrieve_node": "do_retrieve",
            "task_parser_node": "task_parser"
        }
    )
    # do_retrieve执行完成后，进入任务解析节点
    graph.add_edge("do_retrieve", "task_parser")

    # ========= 原有链路（完全不变） =========
    # task_parser执行完成，走条件路由
    graph.add_conditional_edges(
        source="task_parser",
        path=task_router,
        path_map={
            "exec_task": "exec_task",
            "end": "result_collect"
        }
    )

    # exec_task执行完再次走task_router
    graph.add_conditional_edges(
        source="exec_task",
        path=task_router,
        path_map={
            "exec_task": "exec_task",
            "end": "result_collect"
        }
    )

    # 汇总完成直接结束
    graph.add_edge("result_collect", END)

    # 编译图，注入内存checkpoint ✅修复点
    compiled_graph = graph.compile(checkpointer=memory_saver)
    return compiled_graph

# ===================== 自测代码 =====================
if __name__ == "__main__":
    print("==== graph_builder 模块自测 ====")
    from agent.state import AgentState, GraphState
    from utils.helper import generate_trace_id

    # mock一个retriever，自测不需要真实RAG，兼容top_k参数
    class MockRetriever:
        def hybrid_search(self, query, top_k=None):
            return []

    mock_retriever = MockRetriever()
    agent_graph = build_graph(mock_retriever)

    # case1：商品相关query，预期 need_retrieve=True，来源标记 platform_knowledge
    print("\n==== Case1：商品相关问题【简单介绍蓝牙耳机】 ====")
    trace_id = generate_trace_id()
    init_agent_state = AgentState(
        user_query="简单介绍蓝牙耳机",
        trace_id=trace_id
    )
    input_graph_state = {
        "messages": [("user", "简单介绍蓝牙耳机")],
        "agent_state": init_agent_state
    }
    # 新增 config，必须传 thread_id
    config = {"configurable": {"thread_id": "test-thread-001"}}
    result = agent_graph.invoke(input_graph_state, config=config)
    final_agent = result["agent_state"]
    print(f"\n✅执行完成 trace_id={final_agent.trace_id}")
    print(f"need_retrieve: {final_agent.need_retrieve}, answer_source: {final_agent.answer_source}")
    print(f"task_list: {final_agent.task_list}")
    print(f"current_task_index: {final_agent.current_task_index}")
    # 兼容元组 / AIMessage 对象两种格式
    last_msg = result["messages"][-1]
    if isinstance(last_msg, tuple):
        ai_text = last_msg[1]
    elif isinstance(last_msg, AIMessage):
        ai_text = last_msg.content
    else:
        ai_text = str(last_msg)
    print(f"\n📨最终AI消息：{ai_text}")

    # case2：无关业务query，预期 need_retrieve=False，来源标记 llm_knowledge
    print("\n==== Case2：非电商问题【计算机怎么就业】 ====")
    trace_id2 = generate_trace_id()
    init_agent_state2 = AgentState(
        user_query="计算机怎么就业",
        trace_id=trace_id2
    )
    input_graph_state2 = {
        "messages": [("user", "计算机怎么就业")],
        "agent_state": init_agent_state2
    }
    # 新增 config
    config2 = {"configurable": {"thread_id": "test-thread-002"}}
    result2 = agent_graph.invoke(input_graph_state2, config=config2)
    final_agent2 = result2["agent_state"]
    print(f"\n✅执行完成 trace_id={final_agent2.trace_id}")
    print(f"need_retrieve: {final_agent2.need_retrieve}, answer_source: {final_agent2.answer_source}")
    last_msg2 = result2["messages"][-1]
    if isinstance(last_msg2, tuple):
        ai_text2 = last_msg2[1]
    elif isinstance(last_msg2, AIMessage):
        ai_text2 = last_msg2.content
    else:
        ai_text2 = str(last_msg2)
    print(f"\n📨最终AI消息：{ai_text2}")
