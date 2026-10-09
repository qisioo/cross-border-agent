""" LangGraph 全局状态定义 整个 Agent 工作流所有节点共享的数据容器 核心约束：嵌套Pydantic对象禁止原地修改，必须model_copy整体替换返回 messages 统一由外层 GraphState(MessagesState)维护，AgentState不再保存消息 """
from typing import List, Dict, Optional, Any
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    跨境电商运营助手业务状态
    所有节点读写：禁止原地修改属性，使用model_copy(update=xxx)生成新实例返回
    """
    # 用户输入
    user_query: str = Field(description="用户原始自然语言提问")
    original_input: str = Field(default="", description="商品链接/关键词/客服问题等原始输入")
    rewritten_query: str = Field(default="",
                                 description="经过LLM优化后的查询，用于RAG检索和知识库判断，user_query保留用户原始输入")

    # 商品全局复用数据（只抓取一次）
    product_raw_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="抓取到的商品原始信息，包含标题、价格、卖点、评论等"
    )
    product_fetch_completed: bool = Field(default=False, description="标记商品信息是否已经完成抓取，用于多任务复用，避免重复爬虫")

    # 各任务输出结果
    competitor_report: str = Field(default="", description="竞品分析报告文本")
    listing_content: Optional[Dict[str, Any]] = Field(
        default=None,
        description="生成的Listing结构化内容：title、bullet_points、description、ad_copy"
    )
    customer_qa_result: Optional[Dict[str, Any]] = Field(default=None, description="客服回复结果")
    sales_data: Optional[Dict[str, Any]] = Field(default=None, description="销售数据查询结果")

    # 任务调度控制
    task_list: List[str] = Field(default_factory=list, description="LLM拆解出的子任务列表")
    current_task_index: int = Field(default=0, description="当前执行的子任务索引")
    current_task_type: str = Field(default="", description="当前任务类型：analysis/listing/customer_service/sales")
    retry_count: int = Field(default=0, description="当前子任务重试计数")

    # 异常与校验
    error_msg: str = Field(default="", description="异常错误信息")
    format_check_passed: bool = Field(default=False, description="输出格式校验是否通过")

    # 新增优化字段
    conversation_summary: str = Field(default="", description="长对话摘要，用于控制token消耗")
    trace_id: str = Field(default="", description="会话追踪ID，用于日志、链路排查")

    # ==========【新增边界判断相关字段】==========
    retrieved_chunks: Optional[Any] = Field(default=None, description="前置RAG召回的知识库文本块")
    hit_boundary: bool = Field(default=False, description="是否命中知识库边界禁止规则")
    boundary_answer: str = Field(default="", description="边界兜底回复文本")

    # ==========【本次新增：知识检索判断 + 回答来源标记 + final_answer】==========
    need_retrieve: bool = Field(default=False, description="LLM判定：是否需要检索电商私有知识库")
    answer_source: str = Field(default="", description="回答来源标记，可选：platform_knowledge / llm_knowledge")
    final_answer: str = Field(default="", description="Agent汇总后最终回答")


# 外层LangGraph真正使用的状态：继承MessagesState，messages放在顶层
class GraphState(MessagesState):
    """
    LangGraph顶层状态
    messages：顶层字段，由MessagesState原生reducer做消息追加
    agent_state：嵌套业务Pydantic对象，更新必须整体替换
    """
    agent_state: AgentState


# ===================== 内嵌自测代码 =====================
if __name__ == "__main__":
    print("==== 更新后 AgentState & GraphState 自测 ====")
    from langchain_core.messages import HumanMessage

    # 1.实例业务状态
    old_agent = AgentState(
        user_query="帮我分析蓝牙耳机",
        trace_id="trace-20260908-001"
    )
    print(f"初始 trace_id: {old_agent.trace_id}")
    print(f"初始 product_raw_info: {old_agent.product_raw_info}")
    print(f"初始 conversation_summary: {old_agent.conversation_summary}")
    print(f"初始 retry_count: {old_agent.retry_count}")
    print(f"初始 need_retrieve: {old_agent.need_retrieve}")
    print(f"初始 answer_source: {old_agent.answer_source}")
    print(f"初始 final_answer: {old_agent.final_answer}")

    # 模拟节点正确写法：model_copy生成新对象，禁止原地赋值
    mock_product = {"title": "Wireless Earbuds", "price": 29.99}
    new_agent = old_agent.model_copy(update={
        "product_raw_info": mock_product,
        "conversation_summary": "用户询问蓝牙耳机相关运营问题",
        "need_retrieve": True,
        "answer_source": "platform_knowledge",
        "final_answer": "这里是最终回答测试文本"
    })

    print(f"\nmodel_copy之后 product_raw_info title: {new_agent.product_raw_info['title']}")
    print(f"model_copy之后 conversation_summary: {new_agent.conversation_summary}")
    print(f"model_copy之后 need_retrieve: {new_agent.need_retrieve}")
    print(f"model_copy之后 answer_source: {new_agent.answer_source}")
    print(f"model_copy之后 final_answer: {new_agent.final_answer}")

    # 模拟GraphState顶层结构
    graph_state = GraphState(
        messages=[HumanMessage(content="帮我分析蓝牙耳机")],
        agent_state=new_agent
    )

    print(f"\ngraph_state顶层messages数量：{len(graph_state['messages'])}")
    print(f"graph_state['agent_state'].trace_id = {graph_state['agent_state'].trace_id}")
    print(f"graph_state['agent_state'].answer_source = {graph_state['agent_state'].answer_source}")
    print(f"graph_state['agent_state'].final_answer = {graph_state['agent_state'].final_answer}")

    print("\n✅ state模块自测完成，记住规则：嵌套agent_state不能原地修改，必须model_copy整体替换返回")
