""" LangGraph 构图模块 graph_builder.py 组装全部节点、条件边，返回可编译运行的Agent图实例
外部 main.py 需要传入全局的 hybrid_retriever (RAG混合检索对象)
统一链路工作流（方案1：单意图/复合意图完全同构）：
pre_rag_retrieve → intent_split(切分+补全+分组+business) → reflection_check(反思校验)
    → intent_execute(统一执行：按 type/business 路由到客服/销量/竞品/listing/抓取/MCP天气时间/闲聊)
    → result_collect(统一出口：结合原句+子结果，大模型整合，输出校验) → END
所有输入都走同一条链路，不再分"复合/单意图"两条路。
"""
from typing import Any
from functools import partial
from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek
from agent.state import GraphState
from langgraph.checkpoint.memory import MemorySaver
from utils.helper import update_agent_state, generate_trace_id

from agent.nodes import (
    pre_rag_retrieve_node,
    intent_split_node,
    reflection_check_node,
    route_after_split,
    exec_intent_node,
    result_collect_node,
    output_reflection_node,
    route_after_output_reflection,
)

from agent.tools import (
    product_info_fetch,
    competitor_analysis,
    listing_generate,
    sales_data_query,
    customer_service_qa
)

def build_graph(tenant_registry):
    """
    构建并返回编译后的 StateGraph 实例
    :param tenant_registry: 租户检索注册表(rag.tenant_registry.TenantRetrieverRegistry)，main.py创建传入
                            执行节点按 tenant_id+conversation_id 取对应对话检索器，实现多租户知识库隔离
    :return: compiled_graph
    """
    memory_saver = MemorySaver()
    graph = StateGraph(GraphState)

    # ========= 注册节点（统一链路） =========
    graph.add_node("pre_rag_retrieve", pre_rag_retrieve_node)
    graph.add_node("intent_split", intent_split_node)
    graph.add_node("reflection_check", reflection_check_node)
    graph.add_node("intent_execute", partial(exec_intent_node, tenant_registry=tenant_registry))
    graph.add_node("result_collect", result_collect_node)
    graph.add_node("output_reflection", output_reflection_node)

    # ========= 设置入口点【启动首先运行 pre_rag_retrieve】 =========
    graph.set_entry_point("pre_rag_retrieve")

    # ========= 统一链路：pre_rag_retrieve → intent_split → reflection_check → intent_execute → result_collect → output_reflection =========
    graph.add_edge("pre_rag_retrieve", "intent_split")
    graph.add_edge("intent_split", "reflection_check")

    # 反思后统一路由：所有意图（单/复合）都走 intent_execute
    graph.add_conditional_edges(
        source="reflection_check",
        path=route_after_split,
        path_map={
            "intent_execute": "intent_execute"
        }
    )

    # 统一执行完成后，进入统一结果汇总
    graph.add_edge("intent_execute", "result_collect")

    # 汇总后进入输出反思校验
    graph.add_edge("result_collect", "output_reflection")

    # 输出反思后路由：通过则结束；不通过且未超次数则回 result_collect 重新整合
    graph.add_conditional_edges(
        source="output_reflection",
        path=route_after_output_reflection,
        path_map={
            "result_collect": "result_collect",
            "__end__": END
        }
    )

    # 编译图，注入内存checkpoint
    compiled_graph = graph.compile(checkpointer=memory_saver)
    return compiled_graph


# ===================== 自测代码 =====================
if __name__ == "__main__":
    print("==== graph_builder 模块自测 ====")
    from agent.state import AgentState, GraphState
    from utils.helper import generate_trace_id
    from langchain_core.messages import AIMessage

    # mock一个租户注册表，自测不需要真实RAG
    class MockRetriever:
        def hybrid_search(self, query, top_k=None):
            return []

    class MockRegistry:
        def get(self, tenant_id):
            return MockRetriever()
        def get_for(self, tenant_id, conversation_id):
            return MockRetriever()

    mock_registry = MockRegistry()
    agent_graph = build_graph(mock_registry)

    # case1：商品相关query
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
    config = {"configurable": {"thread_id": "test-thread-001"}}
    result = agent_graph.invoke(input_graph_state, config=config)
    final_agent = result["agent_state"]
    print(f"\n✅执行完成 trace_id={final_agent.trace_id}")
    print(f"answer_source: {final_agent.answer_source}")
    print(f"intent_list: {final_agent.intent_list}")
    last_msg = result["messages"][-1]
    if isinstance(last_msg, AIMessage):
        ai_text = last_msg.content
    else:
        ai_text = str(last_msg)
    print(f"\n📨最终AI消息：{ai_text}")

    # case2：非电商query
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
    config2 = {"configurable": {"thread_id": "test-thread-002"}}
    result2 = agent_graph.invoke(input_graph_state2, config=config2)
    final_agent2 = result2["agent_state"]
    print(f"\n✅执行完成 trace_id={final_agent2.trace_id}")
    print(f"answer_source: {final_agent2.answer_source}")
    last_msg2 = result2["messages"][-1]
    if isinstance(last_msg2, AIMessage):
        ai_text2 = last_msg2.content
    else:
        ai_text2 = str(last_msg2)
    print(f"\n📨最终AI消息：{ai_text2}")
