""" Agent全部业务节点集合 所有节点统一入参 state:GraphState，异常捕获，使用update_agent_state更新嵌套状态 禁止原地修改 agent_state 属性 """
import json
from typing import List, Literal,Any
from langchain_core.messages import HumanMessage,AIMessage
from langchain_deepseek import ChatDeepSeek
from agent.state import GraphState, AgentState
from utils.helper import update_agent_state, safe_truncate_text, generate_trace_id
from agent.output_parsers import safe_parse_json
from agent.tools import product_info_fetch, competitor_analysis, listing_generate, sales_data_query, customer_service_qa



MAX_TASK_RETRY = 2  # 单个子任务最大重试次数

def boundary_judge_node(state: GraphState):
    """
    LLM边界判断节点：读取前置RAG召回的chunk，判断是否命中边界规则
    ✅规则更新：
    1. 只有知识库原文明确标注【无法回答/超出能力】这类兜底语句，才触发拦截，直接返回boundary_answer
    2. 知识库没有相关内容 → 不拦截，放行到task_parser_node，由LLM完成任务生成
    ✅新增：同一会话追问（已有历史任务）直接放行，走完整任务编排
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    query = agent.user_query
    chunks = agent.retrieved_chunks
    print(f"[boundary_judge_node][trace_id={trace_id}] 开始边界规则判断")

    # ========== 方案A：硬关键词前置拦截【销量预测类】，优先执行 ==========
    block_keywords = {"预测销量", "下月销量", "下个月销量", "预估销量"}
    for kw in block_keywords:
        if kw in query:
            print(f"[boundary_judge_node][trace_id={trace_id}] 命中硬拦截关键词:{kw}")
            return update_agent_state(
                state,
                hit_boundary=True,
                boundary_answer="该问题超出当前知识库能力范围，请咨询运营专员。"
            )

    # ========== 新增：同一会话追问直接放行，走完整任务编排 ==========
    if agent.task_list and agent.original_input and agent.original_input != query:
        print(f"[boundary_judge_node][trace_id={trace_id}] 检测到同会话追问（历史任务列表非空），跳过知识库边界拦截，放行进入完整任务编排")
        return update_agent_state(
            state,
            hit_boundary=False,
            boundary_answer=""
        )

    # 拼接召回文本，兼容字符串 / LangChain Document 对象
    if not chunks:
        # 没有召回任何知识库内容，直接放行，交给LLM处理
        print(f"[boundary_judge_node][trace_id={trace_id}] 召回文档为空，放行，由LLM原生知识生成回答")
        return update_agent_state(state, hit_boundary=False, boundary_answer="")

    chunk_text_list = []
    for item in chunks:
        if isinstance(item, str):
            chunk_text_list.append(item)
        elif isinstance(item, dict):
            # checkpoint反序列化后，Document变成dict，读取page_content键
            chunk_text_list.append(item.get("page_content", ""))
        else:
            # LangChain Document对象
            chunk_text_list.append(item.page_content)
    context = "\n".join(chunk_text_list)

    # ========== 重写Prompt，修复之前逻辑缺陷 ==========
    prompt = f"""你是边界规则判断器。
任务：判断【参考知识库文本】是否存在明确的兜底声明，说明该类问题无法回答。
用户问题：{query}
参考知识库文本：{context}
规则：
1. 只有知识库原文明确写了【无法回答、超出能力范围，请咨询运营专员】这类兜底语句 → flag=true，reply摘抄知识库原文兜底语句。
2. 知识库没有相关内容，或者知识库内容和问题无关，但**没有兜底声明** → flag=false，reply=""，放行，交给大模型处理。
3. ⚠️重要：不能仅仅因为知识库里面没有对应答案，就拦截问题。
4. reply字段只能摘抄知识库原文，禁止自己编造话术。
5. 只输出标准JSON字符串，不要任何解释、前言、markdown、多余文字。

输出示例1（命中边界）：
{{"flag":true,"reply":"该问题超出当前知识库能力范围，请咨询运营专员。"}}

输出示例2（正常可回答，放行）：
{{"flag":false,"reply":""}}
"""
    hit_flag = False
    reply_text = ""
    max_retry = 2
    for attempt in range(max_retry):
        try:
            llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
            resp = llm.invoke(prompt)
            raw_text = resp.content.strip()
            data = safe_parse_json(raw_text, trace_id)
            hit_flag = bool(data.get("flag", False))
            reply_text = str(data.get("reply", ""))
            break
        except Exception as e:
            print(f"[boundary_judge_node][trace_id={trace_id}] LLM边界判断 第{attempt+1}次解析异常：{str(e)}")
            if attempt >= max_retry - 1:
                # 重试耗尽，默认放行
                hit_flag = False
                reply_text = ""

    return update_agent_state(
        state,
        hit_boundary=hit_flag,
        boundary_answer=reply_text
    )

def task_parser_node(state: GraphState) -> dict[str, Any]:
    """
    LLM版本任务拆解节点
    根据用户query自动识别业务类型，输出结构化子任务列表与task_type
    使用safe_parse_json做LLM输出清洗与JSON校验
    LangGraph节点：入参只有state，返回状态更新字典
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    user_query = agent.user_query
    print(f"[task_parser_node][trace_id={trace_id}] 用户query={user_query}, 开始LLM任务拆解")

    prompt = """
    你是任务拆解专家，请基于当前用户提问分析，输出严格JSON，禁止多余文字。可选业务类型： analysis：竞品分析；customer_service：知识库客服问答；listing：商品Listing文案生成；sales：销售数据查询；product_fetch：抓取商品网页信息。
    输出格式：
    {
        "task_list": ["子任务1","子任务2"],
        "current_task_type": "analysis|customer_service|listing|sales|product_fetch"
    }
    规则：
    1. 用户询问商品FAQ、参数说明、商品概况、爆款情况、笼统销售表现，选customer_service，任务列表仅1项：["知识库检索并生成客服答案"]
    2. 用户要求竞品对比、市场分析，选analysis，拆分成多步分析子任务
    3. 用户需要商品标题、五点描述Listing文案，选listing
    4. 用户查询CSV结构化销售数据选sales：**必须同时明确出现SKU/商品编号 AND 具体年月（例如2026‑08、2026年8月）两个要素；仅商品名称、只问销售情况、没有SKU编号或没有具体月份，禁止选sales，归customer_service**
    5. 用户传入商品URL，要求抓取商品信息，选product_fetch，任务列表仅1项：["抓取商品网页信息"]
    6. 用户纯闲聊、记忆类提问（例如记住名字、简单对话），选customer_service，任务列表仅1项：["对话问答，基于对话历史生成回答"]
    """
    # =====【修改】拆解任务只基于当前用户问题，不传完整对话历史=====
    # 原因：历史中含上一轮超长业务报告会干扰拆解LLM，导致输出非JSON；对话记忆由"对话问答"任务阶段负责
    msg_for_parse = [HumanMessage(content=user_query)]
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    resp = llm.invoke([{"role": "system", "content": prompt}] + msg_for_parse)

    raw_llm_output = resp.content

    # 使用output_parsers解析，自动剥离markdown代码块
    parse_result = safe_parse_json(raw_llm_output, trace_id)

    if parse_result is None:
        # JSON解析失败兜底，降级为客服任务
        print(f"[task_parser_node][trace_id={trace_id}] LLM输出JSON解析失败，启用兜底客服任务")
        task_list = ["知识库检索并生成客服答案"]
        current_task_type = "customer_service"
    else:
        task_list = parse_result.get("task_list", ["知识库检索并生成客服答案"])
        current_task_type = parse_result.get("current_task_type", "customer_service")

    print(f"[task_parser_node][trace_id={trace_id}] 解析任务列表:{task_list}, task_type:{current_task_type}")
    new_agent_state = agent.model_copy(update={
        "task_list": task_list,
        "current_task_type": current_task_type,
        "current_task_index": 0,
        "error_msg": "",
        "retry_count": 0
    })
    return {"agent_state": new_agent_state}


def query_rewrite_node(state: GraphState) -> dict[str, Any]:
    """
    Query改写节点：优化用户提问，提升RAG检索与意图判断准确率
    规则：不改变用户原本诉求，补全模糊信息；闲聊短句不做改写
    输出存入 agent_state.rewritten_query
    """
    agent = state["agent_state"].model_copy()
    trace_id = agent.trace_id
    raw_query = agent.user_query
    print(f"[query_rewrite_node][trace_id={trace_id}] 原始query: {raw_query}")

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    prompt = """ 你是查询优化助手，任务：优化用户提问用于知识库检索。
规则：
1. 禁止修改、曲解用户的核心诉求，不能新增用户没有提到的信息；
2. 用户提问模糊、简写、口语化，补全专业词汇，转为清晰、适合检索的书面问句；
3. 闲聊短句（你好、谢谢、我是谁这类）直接原样返回，不要改写；
4. 只输出改写后的问句，不要额外解释、不要markdown、不要多余文字。

用户原始提问：{raw_query} """
    resp = llm.invoke(prompt.format(raw_query=raw_query))
    rewritten_query = resp.content.strip()
    print(f"[query_rewrite_node][trace_id={trace_id}] 改写后query: {rewritten_query}")
    #兜底，防止返回空字符串
    if not rewritten_query:
        rewritten_query = raw_query

    # =========【缺失核心逻辑：把改写后的字段写入agent_state，生成新state返回】=========
    new_state = update_agent_state(
        state,
        rewritten_query=rewritten_query
    )
    return new_state





def increase_task_index(state: GraphState):
    """
    ✅统一任务下标自增工具，防止业务节点忘记index+1导致无限循环
    封装为独立函数，所有exec任务执行完成后统一调用
    任务成功推进时同步清空error_msg，避免上一次失败的错误残留导致误判
    :param state:
    :return: update_agent_state返回值，直接作为节点返回一部分
    """
    agent = state["agent_state"]
    new_idx = agent.current_task_index + 1
    trace_id = agent.trace_id
    print(f"[increase_task_index][trace_id={trace_id}] 任务下标推进 {agent.current_task_index} → {new_idx}")
    return update_agent_state(
        state,
        current_task_index=new_idx,
        retry_count=0,
        error_msg=""
    )


def task_router(state: GraphState) -> Literal["exec_task", "end"]:
    """
    📌条件路由函数（不是普通节点！）
    逻辑：
        1. 当前子任务重试次数达到MAX_TASK_RETRY上限 → end
        2. 存在error_msg且未达上限、且当前仍有待执行任务 → exec_task（重试当前子任务）
        3. current_task_index < len(task_list) → exec_task 继续执行
        4. 全部子任务完成 → end
    注意：路由函数**不返回状态字典**，只返回路由字符串
    【新增】打印商品缓存标记，用于调试，不改变路由分支
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    total = len(agent.task_list)
    idx = agent.current_task_index
    retry_cnt = agent.retry_count

    # 新增：打印商品缓存状态，方便日志排查，不改变业务流转
    print(
        f"[task_router][trace_id={trace_id}] "
        f"product_fetch_completed={agent.product_fetch_completed}, "
        f"product_raw_info_is_none={agent.product_raw_info is None}"
    )

    # 优先判断：重试次数达到上限 → 结束流程
    if retry_cnt >= MAX_TASK_RETRY:
        print(f"[task_router][trace_id={trace_id}] 任务重试次数{retry_cnt}达到上限{MAX_TASK_RETRY}，终止流程")
        return "end"

    # 检测到错误且未超限、且当前有待执行任务 → 重试当前子任务
    if agent.error_msg.strip() and idx < total:
        print(f"[task_router][trace_id={trace_id}] 检测到错误消息，重试当前子任务 idx={idx} error={agent.error_msg}")
        return "exec_task"

    # 还有未执行任务 → 继续执行
    if idx < total:
        print(f"[task_router][trace_id={trace_id}] 待执行任务 idx={idx}/{total}, task={agent.task_list[idx]}, retry={retry_cnt}")
        return "exec_task"
    else:
        print(f"[task_router][trace_id={trace_id}] 全部子任务执行完毕，结束agent流程")
        return "end"


def pre_rag_retrieve_node(state: GraphState) -> dict[str, Any]:
    """
    前置RAG检索节点
    ✅多轮会话兼容：如果agent_state已存在（从checkpoint加载），继承历史，只更新本次query和trace_id
    不存在则新建AgentState实例
    """
    from utils.helper import generate_trace_id
    from agent.state import AgentState
    trace_id = generate_trace_id()
    last_user_msg = state["messages"][-1]
    user_query = last_user_msg.content
    print(f"[pre_rag_retrieve_node][trace_id={trace_id}] 执行前置知识库检索")

    agent_state = state.get("agent_state")
    if agent_state is None:
        # 首轮对话：初始化全新AgentState
        agent_state = AgentState(
            user_query=user_query,
            trace_id=trace_id
        )
    else:
        # 多轮会话：继承历史状态，仅更新本次用户query与trace_id，保留competitor_report等历史字段
        agent_state = agent_state.model_copy(update={
            "user_query": user_query,
            "trace_id": trace_id
        })

    # ========= 原有前置检索逻辑保持不变 =========
    # 这里保留你原来pre_rag_retrieve_node里的检索逻辑
    # retrieved_chunks = hybrid_retriever.hybrid_search(user_query, top_k=2)
    # agent_state = agent_state.model_copy(update={"retrieved_chunks": retrieved_chunks})

    # ✅直接返回，不再调用update_agent_state，消除警告
    return {"agent_state": agent_state}





def route_after_boundary(state: GraphState) -> Literal["task_parser_node", "result_collect"]:
    """
    边界判断后的条件路由函数（不是普通节点）
    hit_boundary=True：直接跳转到结果汇总，不执行任务拆解
    hit_boundary=False：继续原有流程，进入task_parser_node
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    if agent.hit_boundary:
        print(f"[route_after_boundary][trace_id={trace_id}] 命中知识库边界规则，直接输出兜底回答")
        return "result_collect"
    else:
        print(f"[route_after_boundary][trace_id={trace_id}] 未命中边界，继续执行任务解析")
        return "task_parser_node"

# ========================【本次新增节点与路由，原有代码不动】========================
def need_knowledge_judge_node(state: GraphState):
    """
    LLM业务相关性判断节点
    规则：只要涉及电商商品、店铺产品、跨境商品、店铺销售数据、商品FAQ、售后订单相关，need_retrieve=True，使用平台知识库
    完全无关（天气、计算机就业、机票等）need_retrieve=False，使用大模型原生知识
    写入 need_retrieve 和 answer_source
    """
    agent = state["agent_state"].model_copy()
    trace_id = agent.trace_id
    # 使用改写后的query做判断
    query = agent.rewritten_query
    print(f"[need_knowledge_judge_node][trace_id={trace_id}] 业务相关性判断 rewritten_query={query}")

    judge_prompt = """你是业务检索判断器，严格遵守下面规则：
规则1：用户提问如果属于下面任意一类，need_retrieve=true
- 跨境电商商品、店铺售卖产品、产品参数、规格、材质、使用方法
- 店铺销售数据、商品FAQ、产品评价
- 售后订单类：退款、退货、理赔、物流、发货、清关、发票、质保、配件、平台政策
规则2：用户提问完全和店铺商品/订单无关：天气、计算机就业、机票、历史科普（非店铺商品）等 → need_retrieve=false
规则3：无法确定归属时，保守选择 need_retrieve=true，优先检索知识库
只输出JSON，字段：need_retrieve(bool), reason(str)
禁止额外输出。
"""
    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        # 只传入改写后的query，不再传入历史消息，减少干扰
        resp = llm.invoke(judge_prompt + f"\n用户问题：{query}")
        res = safe_parse_json(resp.content.strip(), trace_id)
        need_retrieve = res.get("need_retrieve", False)
    except Exception as e:
        print(f"[need_knowledge_judge_node][trace_id={trace_id}] 判断LLM调用异常，默认检索知识库：{str(e)}")
        # 异常兜底改成True，防止售后类问题因为LLM解析失败不走知识库
        need_retrieve = True

    source_tag = "platform_knowledge" if need_retrieve else "llm_knowledge"
    # ======修复：update_agent_state第一参数传GraphState(state)，字段用kwargs传递======
    return update_agent_state(
        state,
        need_retrieve=need_retrieve,
        answer_source=source_tag
    )

def do_retrieve_node(state: GraphState, hybrid_retriever):
    """
    知识库检索节点：仅need_retrieve=True才会进入
    调用混合RAG召回文档，存入retrieved_chunks
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    # 改为使用LLM改写后的query进行检索
    rewrite_query = agent.rewritten_query
    print(f"[do_retrieve_node][trace_id={trace_id}] 执行电商知识库混合检索，改写后query={rewrite_query}")

    docs = hybrid_retriever.hybrid_search(rewrite_query, top_k=8)
    new_state = update_agent_state(
        state,
        retrieved_chunks=docs
    )
    return new_state


def route_knowledge_judge(state: GraphState) -> Literal["do_retrieve_node", "task_parser_node"]:
    """知识判断后的条件路由函数"""
    agent = state["agent_state"]
    trace_id = agent.trace_id
    if agent.need_retrieve:
        print(f"[route_knowledge_judge][trace_id={trace_id}] 判断需要检索电商知识库，进入do_retrieve_node")
        return "do_retrieve_node"
    else:
        print(f"[route_knowledge_judge][trace_id={trace_id}] 判断不需要知识库，直接进入任务解析")
        return "task_parser_node"




def exec_task_dispatcher(state: GraphState, hybrid_retriever):
    """
    任务执行分发节点
    根据 agent.current_task_type 分发到不同业务工具：analysis / listing / customer_service / sales / product_fetch
    异常捕获，失败写入 error_msg，用于 task_router 重试
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    current_task = agent.task_list[agent.current_task_index]
    task_type = agent.current_task_type
    print(f"[exec_task_dispatcher][trace_id={trace_id}] 开始执行 idx={agent.current_task_index}, task_name={current_task}, type={task_type}, retry={agent.retry_count}")
    try:
        if task_type == "analysis":
            # ========== 竞品分析任务：先抓一次数据，再并行执行全部子任务，一次完成 ==========
            # 原因：每个子任务独立爬虫会瞬时产生大量并发请求触发站点限流导致超时；
            #       改为共享一次抓取结果，子任务仅并行做LLM分析，既满足"同时执行"又避免重复爬虫
            import concurrent.futures
            from agent.tools import competitor_analysis_parallel, fetch_product_list
            subtasks = list(agent.task_list)
            print(f"[exec_task_dispatcher][trace_id={trace_id}] 竞品分析并行执行 {len(subtasks)} 个子任务")

            # ==========【改动点】从用户query提取英文关键词，构造搜索URL，不再硬编码page-1.html ==========
            # 调用LLM把用户查询翻译成英文搜索关键词，用于books.toscrape搜索
            kw_llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
            kw_prompt = f"""你是关键词提取器，从用户的商品查询语句提取英文搜索关键词，仅输出关键词，不要多余文字。
        用户原始查询：{agent.user_query}
        要求：只输出适合books.toscrape图书网站搜索的英文关键词，如果是耳机/电子产品这类图书网站不存在的品类，直接返回 "none"
        """
            kw_resp = kw_llm.invoke(kw_prompt)
            search_keyword = kw_resp.content.strip()
            if search_keyword.lower() == "none":
                # 品类不在图书网站，直接返回空商品，交给LLM在报告里提示品类不匹配
                list_url = ""
            else:
                # 构造搜索url，关键词做url编码
                import urllib.parse
                encoded_kw = urllib.parse.quote(search_keyword)
                list_url = f"https://books.toscrape.com/catalogue/search?query={encoded_kw}"

            batch_result = {}
            if list_url:
                batch_result = fetch_product_list(list_page_url=list_url, trace_id=trace_id, limit=3)
            else:
                batch_result = {"product_items": [], "warning_msg": "当前查询品类不在图书站点范围内，无商品可抓取"}

            product_items = batch_result.get("product_items", []) if "error" not in batch_result else []
            # ============ 新增代码：提取警告信息 ============
            warning_msg = batch_result.get("warning_msg", "")

            if not product_items:
                # 抓取失败兜底：回退到原competitor_analysis（内部会再次尝试并给出失败提示）
                print(f"[exec_task_dispatcher][trace_id={trace_id}] 竞品数据抓取失败，回退单次竞品分析")
                report = competitor_analysis(
                    product_name=agent.user_query,
                    price_range="",
                    trace_id=trace_id
                )
            else:
                # 第二步：并行执行全部子任务（仅LLM分析，复用同一份抓取数据）
                # ============ 修改函数，增加warning_msg入参 ============
                def _run_analysis(_task):
                    return competitor_analysis_parallel(_task, product_items, trace_id, warning_msg)

                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(3, max(1, len(subtasks)))
                ) as pool:
                    results = list(pool.map(_run_analysis, subtasks))
                report = "\n\n".join(results)

            # 一次性完成全部子任务：下标直接跳到末尾，task_router判定全部完成→end
            new_state = update_agent_state(
                state,
                competitor_report=report,
                current_task_index=len(agent.task_list)
            )
            return new_state

        elif task_type == "product_fetch":
            # =========【新增：商品缓存复用逻辑】=========
            if agent.product_fetch_completed and agent.product_raw_info is not None:
                print(f"[exec_task_dispatcher][trace_id={trace_id}] 商品已抓取完成，跳过爬虫与保存，直接复用缓存product_raw_info")
                # 已有缓存，不执行爬虫，直接推进子任务下标
                return increase_task_index(state)
            # ============================================

            # 修复bug：从用户原始query和改写后query中自动提取http/https链接，解决original_input为空导致爬虫失败的问题
            import re
            url_pattern = re.compile(r'https?://[^\s，。、）)】]+')
            url_match = url_pattern.search(agent.rewritten_query or "") or url_pattern.search(agent.user_query)
            fetch_url = agent.original_input
            if url_match:
                fetch_url = url_match.group(0)
            # 抓取商品信息任务：调用爬虫，写入state，并保存txt到rag/knowledge
            product_info = product_info_fetch(fetch_url, trace_id)
            if product_info.get("error"):
                raise Exception(f"商品抓取失败：{product_info['error']}")
            # 保存商品信息文本到知识库目录
            from agent.tools import save_product_to_knowledge
            save_res = save_product_to_knowledge(
                product_data=product_info,
                save_dir=r"D:\qi_agent\Agent03\rag\knowledge",
                trace_id=trace_id
            )
            if not save_res.get("success"):
                print(f"[exec_task_dispatcher][trace_id={trace_id}] 商品文档保存警告：{save_res.get('error')}")
            # 更新state，存入product_raw_info，标记抓取完成，后续子任务复用
            new_state = update_agent_state(
                state,
                product_raw_info=product_info,
                product_fetch_completed=True
            )

        elif task_type == "listing":
            # Listing文案生成任务：优先复用已经抓取好的product_raw_info
            if agent.product_raw_info is not None:
                product_info = agent.product_raw_info
            else:
                # 无商品信息直接抛出异常，禁止直接拿user_query调用爬虫
                raise Exception("缺少商品基础信息，请先提供商品链接或商品资料")

            listing_result = listing_generate(product_info, "US", trace_id)
            # 新增：捕获listing工具返回的error，抛出异常，走统一重试逻辑
            if listing_result.get("error"):
                raise Exception(f"Listing生成工具异常: {listing_result['error']}")
            new_state = update_agent_state(state, listing_content=listing_result)

        elif task_type == "customer_service":
            # ========== 客服问答分支 ==========
            retrieved_docs = agent.retrieved_chunks or []
            current_task = agent.task_list[agent.current_task_index]
            # 判断：纯对话记忆任务，不走知识库，直接LLM对话
            if current_task == "对话问答，基于对话历史生成回答":
                print(f"[exec_task_dispatcher][trace_id={trace_id}] 纯对话任务，读取完整对话历史生成回答")
                msg_history = state["messages"]
                llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
                system_prompt = "你是跨境电商智能运营助手，根据对话历史自然回答用户，不要输出任何内部标记、子任务信息。"
                resp = llm.invoke([{"role":"system","content":system_prompt}] + msg_history)
                qa_result = {"answer": resp.content}
            else:
                # 普通客服知识库问答，原有逻辑不变
                qa_result = customer_service_qa(
                    question=agent.user_query,
                    retrieve_docs=retrieved_docs,
                    trace_id=trace_id
                )
                if qa_result.get("error"):
                    raise Exception(f"客服问答工具异常: {qa_result['error']}")
            new_state = update_agent_state(state, customer_qa_result=qa_result)

        elif task_type == "sales":
            # 销售数据查询任务：自然语言自动提取SKU+月份
            sales_res = sales_data_query(user_query=agent.user_query, trace_id=trace_id)
            new_state = update_agent_state(state, sales_data=sales_res)

        else:
            raise Exception(f"未知任务类型: {task_type}")

        # 任务执行成功，推进任务下标
        return increase_task_index(new_state)

    except Exception as e:
        err_msg = f"执行任务失败：{str(e)}"
        print(f"[exec_task_dispatcher][trace_id={trace_id}] {err_msg}")
        new_state = update_agent_state(
            state,
            error_msg=err_msg,
            retry_count=agent.retry_count + 1
        )
        return new_state




from langchain_core.messages import AIMessage

def result_collect_node(state: GraphState):
    """
    结果汇总节点：统一交给大模型整理，隐藏所有内部子任务、调试标记
    用户仅获取干净的最终回答，data字段保留原始材料用于后台调试
    【本次修改】按task_type单独选取对应原始素材，防止跨任务内容干扰LLM；原始素材必须经过LLM优化，禁止直接输出
    """
    agent = state["agent_state"].model_copy()
    trace_id = agent.trace_id
    print(f"[result_collect_node][trace_id={trace_id}] 汇总全部业务结果")
    raw_source_data = None
    final_output = {}

    if agent.answer_source == "platform_knowledge":
        source_label = "【基于电商平台知识库回答】"
    else:
        source_label = "【大模型原生知识回答】"

    # ========== 核心改动：按任务类型，只选取当前任务的原始素材，不拼接全部内容 ==========
    task_type = agent.current_task_type

    if agent.hit_boundary and agent.boundary_answer and agent.boundary_answer.strip():
        raw_source_data = agent.boundary_answer
        final_output["boundary_answer"] = agent.boundary_answer
    elif task_type == "product_fetch":
        raw_source_data = agent.product_raw_info
        final_output["product_raw_info"] = agent.product_raw_info
    elif task_type == "listing":
        raw_source_data = agent.listing_content
        final_output["listing_content"] = agent.listing_content
    elif task_type == "analysis":
        raw_source_data = agent.competitor_report
        final_output["competitor_report"] = agent.competitor_report
    elif task_type == "sales":
        raw_source_data = agent.sales_data
        final_output["sales_data"] = agent.sales_data
    elif task_type == "customer_service":
        raw_source_data = agent.customer_qa_result
        final_output["customer_qa"] = agent.customer_qa_result
    else:
        raw_source_data = None

    # 兜底：素材为空/空字符串/空字典，直接返回兜底文案，绝不送入LLM
    if not raw_source_data or (isinstance(raw_source_data, dict) and len(raw_source_data) == 0):
        final_answer = "暂时没有找到相关信息。"
        print(f"[result_collect_node][trace_id={trace_id}] 无参考素材，使用兜底回答")
    else:
        # ==========【Skill规则注入】润色固定约束下沉到skill/rules/result_collect.md，Skill结果优先，失败走既有兜底 ==========
        from utils.skill_loader import get_skill_loader
        # ======================================================================
        user_prompt = f"""当前任务类型：{task_type} 【原始素材】 {raw_source_data} 请基于上面这份原始素材，润色排版，输出面向用户的最终回答。"""
        try:
            result_collect_rule = get_skill_loader().load_rule("result_collect")
            if not result_collect_rule:
                raise Exception("结果汇总润色规则加载失败（skill/rules/result_collect.md缺失）")
            llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
            resp = llm.invoke([
                {"role": "system", "content": result_collect_rule},
                {"role": "user", "content": user_prompt}
            ])
            final_answer = f"{source_label}\n{resp.content.strip()}"
            print(f"[result_collect_node][trace_id={trace_id}] LLM整理后的结果：\n{final_answer}")
        except Exception as e:
            print(f"[result_collect_node][trace_id={trace_id}] LLM文本整理异常: {str(e)}")
            # LLM调用失败或规则缺失兜底，直接原始素材输出，防止链路崩溃
            final_answer = f"{source_label}\n{str(raw_source_data)}"

    # 更新agent_state内部字段final_answer
    updated_state_dict = update_agent_state(
        state,
        final_answer=final_answer
    )
    # 从入参state读取messages，追加标准AIMessage对象
    new_ai_msg = AIMessage(content=final_answer)
    new_messages = state["messages"] + [new_ai_msg]
    # 将更新后的消息列表写回GraphState
    updated_state_dict["messages"] = new_messages
    return updated_state_dict



# ====================================================================================

# ===================== 自测入口 =====================
if __name__ == "__main__":
    print("==== agent.nodes 模块自测 ====")
    from agent.state import AgentState, GraphState
    from utils.helper import generate_trace_id

    trace_id = generate_trace_id()
    init_agent = AgentState(user_query="", trace_id=trace_id)

    # case1 任务解析：竞品分析
    gs1 = GraphState(
        messages=[HumanMessage(content="帮我分析蓝牙耳机的竞品情况")],
        agent_state=init_agent
    )
    out1 = task_parser_node(gs1)
    ag1 = out1["agent_state"]
    print(f"\ncase1 task_list:{ag1.task_list}, type:{ag1.current_task_type}, index:{ag1.current_task_index}, retry_count:{ag1.retry_count}")
    route1 = task_router({"agent_state": ag1, "messages": gs1["messages"]})
    print(f"case1路由输出：{route1}")

    # case2 模拟任务全部跑完：index推进到等于列表长度
    gs2 = GraphState(messages=[], agent_state=ag1)
    out2 = update_agent_state(gs2, current_task_index=len(ag1.task_list))
    ag2 = out2["agent_state"]
    route2 = task_router({"agent_state": ag2, "messages": []})
    print(f"\ncase2全部任务完成后路由输出：{route2}")

    # case3：存在error_msg，重试未达上限，预期返回exec_task
    gs3 = GraphState(messages=[], agent_state=ag1)
    out3 = update_agent_state(gs3, error_msg="执行出错", retry_count=1)
    ag3 = out3["agent_state"]
    route3 = task_router({"agent_state": ag3, "messages": []})
    print(f"case3 存在错误，重试未达上限，路由输出：{route3}")

    # case4：重试次数超限测试
    gs4 = GraphState(messages=[], agent_state=ag1)
    out4 = update_agent_state(gs4, retry_count=MAX_TASK_RETRY)
    ag4 = out4["agent_state"]
    route4 = task_router({"agent_state": ag4, "messages": []})
    print(f"case4 重试达到上限路由输出：{route4}")

    # case5：测试下标自增函数 increase_task_index
    gs5 = GraphState(messages=[], agent_state=ag1)
    out5 = increase_task_index(gs5)
    ag5 = out5["agent_state"]
    print(f"\ncase5 index自增后 old={ag1.current_task_index}, new={ag5.current_task_index}, retry={ag5.retry_count}, error_msg={ag5.error_msg}")



    # =========【新增自测用例：测试知识判断节点】=========
    print("\n==== 新增case7：知识检索判断节点自测 ====")
    # case7-1：电商商品相关问题，预期 need_retrieve=True
    gs7_1 = GraphState(messages=[HumanMessage(content="简单介绍蓝牙耳机")], agent_state=init_agent)
    out7_1 = need_knowledge_judge_node(gs7_1)
    ag7_1 = out7_1["agent_state"]
    print(f"case7-1 query=简单介绍蓝牙耳机 | need_retrieve={ag7_1.need_retrieve}, answer_source={ag7_1.answer_source}")
    route7_1 = route_knowledge_judge(out7_1)
    print(f"case7-1 路由结果：{route7_1}")

    # case7-2：无关业务问题，预期 need_retrieve=False
    gs7_2 = GraphState(messages=[HumanMessage(content="计算机怎么就业")], agent_state=init_agent)
    out7_2 = need_knowledge_judge_node(gs7_2)
    ag7_2 = out7_2["agent_state"]
    print(f"case7-2 query=计算机怎么就业 | need_retrieve={ag7_2.need_retrieve}, answer_source={ag7_2.answer_source}")
    route7_2 = route_knowledge_judge(out7_2)
    print(f"case7-2 路由结果：{route7_2}")
    # ====================================================
    # =========【新增case8：商品URL抓取完整链路测试】=========
    print("\n==== case8：商品URL抓取完整链路测试 ====")
    test_trace_id = generate_trace_id()
    test_agent = AgentState(
        user_query="抓取这个商品页面信息",
        original_input="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        trace_id=test_trace_id
    )
    gs8 = GraphState(messages=[HumanMessage(content="抓取这个商品页面信息")], agent_state=test_agent)
    # 1.任务解析
    out_parser = task_parser_node(gs8)
    ag8 = out_parser["agent_state"]
    print(f"case8 解析结果 task_list:{ag8.task_list}, task_type:{ag8.current_task_type}, index:{ag8.current_task_index}")
    # 2.路由判断
    route8 = task_router({"agent_state": ag8, "messages": gs8["messages"]})
    print(f"case8 路由：{route8}")
    # 3.执行任务分发（传入hybrid_retriever，这里mock传None，仅测试爬虫分支）
    from rag.hybrid_retriever import HybridRetriever
    mock_retriever = HybridRetriever()
    out_exec = exec_task_dispatcher({"agent_state": ag8, "messages": gs8["messages"]}, mock_retriever)
    ag8_after = out_exec["agent_state"]
    print(f"case8 执行完成，product_raw_info:{ag8_after.product_raw_info}")
    print(f"case8 当前任务下标：{ag8_after.current_task_index}")
    route8_after = task_router({"agent_state": ag8_after, "messages": gs8["messages"]})
    print(f"case8 执行后路由：{route8_after}")

    print("\n✅ agent.nodes自测完成")
