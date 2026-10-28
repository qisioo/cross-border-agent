""" Agent全部业务节点集合 所有节点统一入参 state:GraphState，异常捕获，使用update_agent_state更新嵌套状态 禁止原地修改 agent_state 属性 """
import json
# ===== 统一从 .env 加载项目密钥（DEEPSEEK_API_KEY 等），并覆盖系统环境变量旧值 =====
# 目的：无论从哪个入口进入（main.py / streamlit / 模块直接运行），都保证用 .env 的最新密钥，
#       而不是可能残留的系统环境变量旧值（例如已废弃的 key）。
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import List, Literal,Any
from langchain_core.messages import HumanMessage,AIMessage
from langchain_deepseek import ChatDeepSeek
from agent.state import GraphState, AgentState
from utils.helper import update_agent_state, safe_truncate_text, generate_trace_id
from agent.output_parsers import safe_parse_json
from agent.tools import product_info_fetch, competitor_analysis, listing_generate, sales_data_query, customer_service_qa



MAX_TASK_RETRY = 2  # 单个子任务最大重试次数


def _re_match_business(text: str) -> str:
    """
    业务类型辅助判断（方案1内化旧 task_parser 的 business 判断）
    仅对 knowledge 子意图生效，按文本关键词细分为：
        sales(销量) / analysis(竞品) / listing / product_fetch(抓取) / customer_service(客服，默认)
    """
    import re as _re
    _SALES_PAT = _re.compile(r"销量|销售数据|销售额|下月销量|SKU|库存|预测.*销量")
    _ANALYSIS_PAT = _re.compile(r"竞品|分析|对比|市场")
    _LISTING_PAT = _re.compile(r"listing|标题|五点|五点描述|描述|文案")
    _FETCH_PAT = _re.compile(r"抓取|爬|商品链接|URL|商品网页")
    if _SALES_PAT.search(text):
        return "sales"
    if _ANALYSIS_PAT.search(text):
        return "analysis"
    if _LISTING_PAT.search(text):
        return "listing"
    if _FETCH_PAT.search(text):
        return "product_fetch"
    return "customer_service"

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

def intent_split_node(state: GraphState) -> dict[str, Any]:
    """
    【复合意图拆分节点 · 方案B(两层)：切分到原子粒度 + 分组标记 + 补全，不改动原有单意图逻辑】
    作用：把用户一整句话拆成多个"原子级"子意图，每个子意图单一诉求、单一来源，
          对"同属一个上层问题"的子意图打相同 group_id，并结合原句补全省略语义。
    例："武汉今天天气怎么样，给我推荐一个蓝牙耳机，如果我下单的话武汉的天气会不会影响物流运输，大概多少天能到"
        → [
            {"intent":"武汉今天天气怎么样","group":1,"type":"non_knowledge"},
            {"intent":"给我推荐一个蓝牙耳机","group":2,"type":"knowledge"},
            {"intent":"武汉今天的天气如何","group":1,"type":"non_knowledge"},   # 同属"天气是否影响物流"
            {"intent":"大模型推荐的蓝牙耳机物流运输大概几天能到","group":1,"type":"knowledge"},  # 同组，已补全
          ]
    类型判断用"两层"（不用LLM判类型，更稳，也避免过度细分）：
        knowledge      → 命中业务词（商品/物流/售后/销量等），需要读知识库
        non_knowledge  → 其余全部（天气/时间/闲聊/通用知识），不读知识库；
                         是否调MCP工具（天气/时间）由 exec_intent_node 内 detect_tool_need 决定
    单意图或拆分失败时，intent_list 置空，走原有单意图逻辑。
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    query = agent.user_query
    print(f"[intent_split_node][trace_id={trace_id}] 开始复合意图拆分: {query}")

    # 1. 用LLM把整句拆成"原子级子意图"+分组标记+结合原句补全省略
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    prompt = """你是意图拆分器。任务：把用户一句话拆成多个"原子级"子意图，结合原句补全省略语义，并为同属一个上层问题的子意图打上相同分组编号。

规则：
1. 每个子意图必须单一诉求、单一来源、能从原话中直接对应，禁止改写、禁止新增、禁止删除关键信息。
2. 切分到原子粒度：只要原话里隐含了多个可独立获取的信息诉求，就要拆成多个子意图。
   例："天气会不会影响物流、大概几天能到" 隐含"查天气"和"查物流规则"两个诉求，要拆成两个子意图，但它们属于同一个上层问题，group编号相同。
3. 组的概念：如果多个子意图最终要放在一起推理（例如"天气"+"物流规则"一起回答"天气是否影响物流"），它们必须用相同的group编号。
   相互独立、不需要联合推理的子意图，用不同的group编号。
4. 重要（补全省略）：子意图若存在省略，必须结合【用户原句】理解指代关系，补全为主语/宾语，使其能独立理解。
   - 补全只能使用原句中已出现的对象，禁止凭空编造、禁止换对象。
   - 例：原句是"给我推荐一个蓝牙耳机……物流大概多少天能到"，那么省略的"物流运输大概多少天能到"应补全为"【原句推荐的蓝牙耳机】物流运输大概多少天能到"，绝不能补成"牙刷"或"电脑"。
   - 若子意图本身语义完整，无需补全，原样保留。
5. 重要：不要把一句话里的"程度/时间碎片"（例如"大概多少天""多久能到""怎么样"）单独拆成一个子意图。这类碎片必须并入它所属的那个主诉求子意图，不要拆成独立的残缺子意图。
6. group编号从1开始，用整数。
7. 如果整句话是单一意图、单一来源，只输出1个子意图。
8. 只输出严格JSON：{"intents": [{"intent": "子意图1", "group": 1}, {"intent": "子意图2", "group": 1}, ...]}，不要多余文字。

用户原句（补全时必须参考它，理解指代关系）：__USER_QUERY__"""
    try:
        # 用replace注入query，避免prompt内JSON示例花括号与.format冲突
        final_prompt = prompt.replace("__USER_QUERY__", query)
        resp = llm.invoke(final_prompt)
        data = safe_parse_json(resp.content.strip(), trace_id)
        raw_intents = data.get("intents", [])
        print(f"[intent_split_node][trace_id={trace_id}] LLM拆分原始返回: {resp.content[:300]}")
    except Exception as e:
        print(f"[intent_split_node][trace_id={trace_id}] 拆分异常，降级整句: {e}")
        raw_intents = []

    # 兼容两种返回：字符串列表（旧） / 带group的字典列表
    intents = []
    for it in raw_intents:
        if isinstance(it, dict):
            intents.append({"text": str(it.get("intent", "")).strip(), "group": int(it.get("group", 0))})
        elif isinstance(it, str):
            intents.append({"text": it.strip(), "group": 0})

    # 统一链路：不区分单/多意图，全部构造为子意图列表（单意图就是1个），统一走 intent_execute
    # 兜底：完全拆不出任何子意图时，把整句当作1个子意图，保证链路不中断
    if not intents:
        intents = [{"text": agent.user_query.strip(), "group": 1}]

    # 2. 对每个子意图判断类型（两层：业务→knowledge，其余→non_knowledge）
    # 不再细分 weather/chat——它们在"是否读知识库"上同属"不读"，是否调MCP工具由执行节点决定
    # 方案1：knowledge 子意图额外标注 business 类型（sales/analysis/listing/product_fetch/customer_service），
    #         供 exec_intent_node 按业务路由到具体工具，保证销量/竞品/listing/抓取能力全部保留。
    import re as _re
    _KNOWLEDGE_PAT = _re.compile(r"商品|耳机|物流|发货|运输|售后|退货|退款|客服|推荐|价格|优惠|活动|促销|满减|折扣|优惠券|包邮|积分|价保|运费|到货|配送|库存|销量|几天|多久|销售|查询|分析|listing|标题|五点|描述")
    _TIME_PAT_EARLY = _re.compile(r"几点|什么时间|现在时间|当前时间|几点了|时区|几月几号|什么时候|几点钟|时间是多少|现在几点|此刻时间|日期|几号|星期几|周几")
    intent_items = []
    for it in intents:
        text = it["text"]
        if not text:
            continue
        # 时间/日期类问题优先走 non_knowledge，避免被"几天|多久"等业务词误命中
        if _TIME_PAT_EARLY.search(text):
            itype = "non_knowledge"
            business = ""
        elif _KNOWLEDGE_PAT.search(text):
            itype = "knowledge"
            # 内化旧 task_parser 的业务类型判断，标注具体业务（销量/竞品/listing/抓取/客服）
            business = _re_match_business(text)
        else:
            itype = "non_knowledge"
            business = ""
        intent_items.append({"intent": text, "type": itype, "group_id": it["group"], "business": business})

    print(f"[intent_split_node][trace_id={trace_id}] 统一拆分为 {len(intent_items)} 个子意图: {intent_items}")
    return update_agent_state(state, intent_list=intent_items, sub_results=[], sub_executed=False)


def reflection_check_node(state: GraphState) -> dict[str, Any]:
    """
    【反思校验节点 · 增量新增】
    在意图切分后、逐意图执行前，让大模型审视切分结果是否正确：
      1. 每个子意图是否都能在原句中找到对应（没编造）？
      2. 补全部分是否曲解了原意、是否换成了原句没有的对象（没歪曲）？
      3. 分组是否合理（同属一个上层问题的有没有归错组）？
    有问题 → 返回修正后的 intent_list；没问题 → 原样保留。
    反思失败时，保留原切分结果（不阻断链路），仅记录日志。
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    if not agent.intent_list:
        return update_agent_state(state, intent_list=agent.intent_list, sub_results=[], sub_executed=False)

    original_query = agent.user_query
    current_intents = agent.intent_list
    print(f"[reflection_check_node][trace_id={trace_id}] 开始反思校验 {len(current_intents)} 个子意图")

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    # 把当前切分结果序列化给LLM审视
    current_json = json.dumps(
        [{"intent": it["intent"], "group": it["group_id"], "type": it["type"]} for it in current_intents],
        ensure_ascii=False
    )
    reflection_prompt = """你是意图切分评审员。任务：审视下面这份"切分结果"是否正确，若有问题则修正后返回，若无问题则原样返回。

【用户原句】{orig}
【当前切分结果】{current}

检查三点：
1. 是否编造：每个子意图是否都能在用户原句中找到对应内容？（不能用原句没有的对象）
2. 是否曲解/换对象：补全部分是否把主语/宾语换成了原句没有的对象？（例如原句是"蓝牙耳机"，补全成"牙刷/电脑"就是错误）
3. 分组是否合理：同属一个上层问题、需要联合推理的子意图，是否用了相同的group编号？相互独立的子意图是否用了不同编号？

如果完全正确，原样返回。
如果有问题，输出修正后的完整切分结果。

只输出严格JSON：{"intents": [{"intent": "子意图1", "group": 1, "type": "knowledge|non_knowledge"}, ...]}，不要多余文字。
不要删除有用的子意图，不要新增原句没有的诉求。"""
    try:
        final_prompt = reflection_prompt.replace("{orig}", original_query).replace("{current}", current_json)
        resp = llm.invoke(final_prompt)
        data = safe_parse_json(resp.content.strip(), trace_id)
        raw_reflected = data.get("intents", [])
        if raw_reflected:
            # 重新构造意图列表（保留原有类型与分组，若LLM给出type则采用）
            new_items = []
            for it in raw_reflected:
                if isinstance(it, dict):
                    text = str(it.get("intent", "")).strip()
                    gid = int(it.get("group", 0))
                    itype = str(it.get("type", "")).strip()
                    if not text:
                        continue
                    if itype not in ("knowledge", "non_knowledge"):
                        # 兼容旧的 weather/time/chat 标记，统一映射到两层
                        itype = "knowledge" if itype == "knowledge" else "non_knowledge"
                    # 保留原 business 标注（切分时已推断；反思仅审视切分正确性，不重算业务类型）
                    business = ""
                    if itype == "knowledge":
                        # 从原切分中找回 business
                        for _orig in current_intents:
                            if _orig.get("intent") == text:
                                business = _orig.get("business", "")
                                break
                        if not business:
                            # 找不到原business（反思改了文本），按当前文本快速重判
                            _b = _re_match_business(text)
                            business = _b
                    new_items.append({"intent": text, "type": itype, "group_id": gid, "business": business})
            if new_items and len(new_items) >= 1:
                print(f"[reflection_check_node][trace_id={trace_id}] 反思校验通过/已修正，共 {len(new_items)} 个子意图")
                return update_agent_state(state, intent_list=new_items, sub_results=[], sub_executed=False)
    except Exception as e:
        print(f"[reflection_check_node][trace_id={trace_id}] 反思校验异常，保留原切分: {e}")

    # 兜底：反思失败，保留原切分结果
    print(f"[reflection_check_node][trace_id={trace_id}] 反思校验未产出修正，保留原切分")
    return update_agent_state(state, intent_list=agent.intent_list, sub_results=[], sub_executed=False)


def route_after_split(state: GraphState) -> Literal["intent_execute"]:
    """统一链路路由：所有意图（单/复合）统一走 intent_execute，由执行节点按 type/business 分发。
    复合意图逐个子意图执行；单意图就是1个子意图，同样走统一执行。"""
    agent = state["agent_state"]
    print(f"[route_after_split][trace_id={agent.trace_id}] 统一链路，进入逐子意图执行（共{len(agent.intent_list) or 0}个）")
    return "intent_execute"


def exec_intent_node(state: GraphState, tenant_registry) -> dict[str, Any]:
    """
    【逐子意图执行节点 · 增量新增】
    对 intent_list 里每个子意图独立执行：
        weather/time → 调 MCP 工具取实时数据
        knowledge    → 知识库检索 + 客服问答
        chat         → 纯对话（LLM + 对话历史）
    每个子意图的结果收集进 sub_results，最后统一交给 result_collect 衔接推理。
    任一子意图失败：记录错误，不中断整条链路。
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    results = []
    print(f"[exec_intent_node][trace_id={trace_id}] 开始逐个执行 {len(agent.intent_list)} 个子意图")

    from agent.mcp_tools import detect_tool_need, get_weather, get_time

    for item in agent.intent_list:
        itype = item.get("type")
        intent = item.get("intent", "")
        group_id = item.get("group_id", 0)
        business = item.get("business", "")
        try:
            if itype == "knowledge":
                # ========== 业务类：按 business 路由到具体业务工具（方案1内化旧 task_parser 能力） ==========
                # 销量/竞品/listing/抓取/客服 全部保留，子意图独立执行，不依赖旧 task_parser/exec_task 链路
                if business == "sales":
                    # 销售数据查询：子意图文本已补全（如"查询SKU001在2026-09的销售数据"），直接交给sales_data_query
                    sales_res = sales_data_query(user_query=intent, trace_id=trace_id)
                    ans = sales_res.get("data", "") if not sales_res.get("error") else f"该子意图暂无法回答"
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[sales][组{group_id}] 销量查询: {intent} → {str(ans)[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": str(ans)})

                elif business == "analysis":
                    # 竞品分析：复用旧链路的并行分析能力（抓一次数据，并行LLM分析全部子任务）
                    import concurrent.futures
                    from agent.tools import competitor_analysis_parallel, fetch_product_list
                    subtasks = [intent]  # 单个子意图即一个分析任务
                    kw_llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
                    kw_prompt = f"""你是关键词提取器，从用户的商品查询语句提取英文搜索关键词，仅输出关键词，不要多余文字。
                用户原始查询：{intent}
                要求：只输出适合books.toscrape图书网站搜索的英文关键词，如果是耳机/电子产品这类图书网站不存在的品类，直接返回 "none"
                """
                    kw_resp = kw_llm.invoke(kw_prompt)
                    search_keyword = kw_resp.content.strip()
                    if search_keyword.lower() == "none":
                        list_url = ""
                    else:
                        import urllib.parse
                        encoded_kw = urllib.parse.quote(search_keyword)
                        list_url = f"https://books.toscrape.com/catalogue/search?query={encoded_kw}"
                    batch_result = {}
                    if list_url:
                        batch_result = fetch_product_list(list_page_url=list_url, trace_id=trace_id, limit=3)
                    else:
                        batch_result = {"product_items": [], "warning_msg": "当前查询品类不在图书站点范围内，无商品可抓取"}
                    product_items = batch_result.get("product_items", []) if "error" not in batch_result else []
                    warning_msg = batch_result.get("warning_msg", "")
                    if not product_items:
                        report = competitor_analysis(product_name=intent, price_range="", trace_id=trace_id)
                    else:
                        def _run_analysis(_task):
                            return competitor_analysis_parallel(_task, product_items, trace_id, warning_msg)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            report_list = list(pool.map(_run_analysis, subtasks))
                        report = "\n\n".join(report_list)
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[analysis][组{group_id}] 竞品分析: {intent} → {report[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": report})

                elif business == "listing":
                    # Listing文案生成：需要商品基础信息（从缓存取或要求提供链接）
                    if agent.product_raw_info is not None:
                        product_info = agent.product_raw_info
                    else:
                        # 无商品信息：交给客服问答兜底（提示需要商品资料），不抛异常阻断
                        retriever = tenant_registry.get_for(agent.tenant_id, agent.conversation_id)
                        docs = retriever.hybrid_search(intent, top_k=4) if retriever is not None else []
                        qa = customer_service_qa(question=intent, retrieve_docs=docs, trace_id=trace_id)
                        ans = qa.get("answer", "") if not qa.get("error") else f"该子意图暂无法回答"
                        print(f"[exec_intent_node][trace_id={trace_id}] 子意图[listing][组{group_id}] 缺商品资料，回退客服: {intent} → {ans[:80]}")
                        results.append({"type": itype, "intent": intent, "group_id": group_id, "result": ans})
                        continue
                    listing_result = listing_generate(product_info, "US", trace_id)
                    if listing_result.get("error"):
                        raise Exception(f"Listing生成工具异常: {listing_result['error']}")
                    ans = (
                        f"标题：{listing_result.get('title', '')}\n"
                        f"五点描述：{listing_result.get('bullet_points', '')}\n"
                        f"广告文案：{listing_result.get('ad_copy', '')}"
                    )
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[listing][组{group_id}] Listing生成: {intent} → {str(ans)[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": str(ans)})

                elif business == "product_fetch":
                    # 商品网页抓取：从子意图文本提取URL/关键词，调用爬虫
                    import re as _re
                    url_pattern = _re.compile(r'https?://[^\s，。、）)】]+')
                    url_match = url_pattern.search(intent) or url_pattern.search(agent.user_query)
                    fetch_url = agent.original_input
                    if url_match:
                        fetch_url = url_match.group(0)
                    if not fetch_url:
                        # 没有URL：当商品关键词客服问答处理，不抛异常
                        retriever = tenant_registry.get_for(agent.tenant_id, agent.conversation_id)
                        docs = retriever.hybrid_search(intent, top_k=4) if retriever is not None else []
                        qa = customer_service_qa(question=intent, retrieve_docs=docs, trace_id=trace_id)
                        ans = qa.get("answer", "") if not qa.get("error") else f"该子意图暂无法回答"
                        print(f"[exec_intent_node][trace_id={trace_id}] 子意图[product_fetch][组{group_id}] 无URL，回退客服: {intent} → {ans[:80]}")
                        results.append({"type": itype, "intent": intent, "group_id": group_id, "result": ans})
                        continue
                    product_info = product_info_fetch(fetch_url, trace_id)
                    if product_info.get("error"):
                        raise Exception(f"商品抓取失败：{product_info['error']}")
                    from agent.tools import save_product_to_knowledge
                    save_res = save_product_to_knowledge(product_data=product_info, save_dir=r"D:\qi_agent\Agent03\rag\knowledge", trace_id=trace_id)
                    if not save_res.get("success"):
                        print(f"[exec_intent_node][trace_id={trace_id}] 商品文档保存警告：{save_res.get('error')}")
                    # 更新到state供后续listing等复用（不在此节点改state，仅记录结果）
                    product_summary = product_info.get("title", "") if isinstance(product_info, dict) else str(product_info)
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[product_fetch][组{group_id}] 商品抓取: {intent} → {str(product_summary)[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": f"已抓取商品信息：{product_summary}"})

                else:
                    # customer_service（默认）：知识库检索 + 客服问答（多租户隔离）
                    retriever = tenant_registry.get_for(agent.tenant_id, agent.conversation_id)
                    if retriever is None:
                        docs = []
                    else:
                        docs = retriever.hybrid_search(intent, top_k=4)
                    qa = customer_service_qa(question=intent, retrieve_docs=docs, trace_id=trace_id)
                    ans = qa.get("answer", "") if not qa.get("error") else f"该子意图暂无法回答"
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[客服][组{group_id}] 知识库: {intent} → {ans[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": ans})

            else:
                # 非业务类：不读知识库。内部判断——命中天气/时间则调MCP工具取实时数据，否则直接对话
                tool_need = detect_tool_need(intent)
                if tool_need is not None and tool_need[0] == "weather":
                    tool_result = get_weather(tool_need[1].get("city", "武汉"))
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[天气][组{group_id}] 调MCP: {intent} → {tool_result}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": tool_result})
                elif tool_need is not None and tool_need[0] == "time":
                    tool_result = get_time()
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[时间][组{group_id}] 调MCP: {intent} → {tool_result}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": tool_result})
                else:
                    # 纯对话：LLM + 对话历史（天气之外的非业务问题如"计算机就业"、闲聊都走这里，DeepSeek原生知识即可回答）
                    msg_history = state["messages"]
                    chat_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
                    resp = chat_llm.invoke(
                        [{"role": "system", "content": "你是跨境电商智能运营助手，根据对话历史自然回答用户，不要输出内部标记。"}]
                        + msg_history
                    )
                    print(f"[exec_intent_node][trace_id={trace_id}] 子意图[对话][组{group_id}] 闲聊: {intent} → {resp.content[:80]}")
                    results.append({"type": itype, "intent": intent, "group_id": group_id, "result": resp.content})

        except Exception as e:
            print(f"[exec_intent_node][trace_id={trace_id}] 子意图[{itype}] 执行异常: {e}")
            results.append({"type": itype, "intent": intent, "group_id": group_id, "result": f"该子意图处理失败"})

    return update_agent_state(
        state,
        sub_results=results,
        sub_executed=True
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

def do_retrieve_node(state: GraphState, tenant_registry):
    """
    知识库检索节点：仅need_retrieve=True才会进入
    按"企业+对话"(tenant_id + conversation_id) 从注册表取本对话的检索器，调用混合RAG召回文档存入retrieved_chunks
    对话级隔离：检索只查本对话绑定的知识库，绝不跨对话；该对话未构建知识库时返回空检索（纯隔离）
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    tenant_id = agent.tenant_id
    conversation_id = agent.conversation_id
    # 改为使用LLM改写后的query进行检索
    rewrite_query = agent.rewritten_query
    # get_for：conversation_id 非空则只取本对话检索器 key=tenant::conv，空则取企业级 key=tenant（兼容旧接口）
    retriever = tenant_registry.get_for(tenant_id, conversation_id)
    if retriever is None:
        print(f"[do_retrieve_node][trace_id={trace_id}] 对话 {tenant_id}::{conversation_id} 未构建知识库，返回空检索（不跨对话）")
        return update_agent_state(state, retrieved_chunks=[])
    print(f"[do_retrieve_node][trace_id={trace_id}] 对话 {tenant_id}::{conversation_id} 执行知识库混合检索，改写后query={rewrite_query}")

    docs = retriever.hybrid_search(rewrite_query, top_k=8)
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




def exec_task_dispatcher(state: GraphState, tenant_registry):
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

                # =====【新增：MCP 天气/时间能力，仅在日常对话中生效】=====
                # 用 agent/mcp_tools.py 检测用户这句话是否需要天气/时间工具；
                # 需要则通过 MCP（独立服务器 mcp_demo/server.py）拿到实时结果，
                # 把结果塞给 LLM 让它自然回复。工具实现与 Agent 解耦，体现 MCP 复用。
                tool_need = None
                try:
                    from agent.mcp_tools import detect_tool_need, get_weather, get_time
                    tool_need = detect_tool_need(agent.user_query)
                except Exception as mcp_e:
                    print(f"[exec_task_dispatcher][trace_id={trace_id}] MCP检测异常: {mcp_e}")
                    tool_need = None

                if tool_need:
                    tool_name = tool_need[0]
                    tool_args = tool_need[1]
                    if tool_name == "weather":
                        tool_result = get_weather(tool_args.get("city", "武汉"))
                    else:
                        tool_result = get_time()
                    print(f"[exec_task_dispatcher][trace_id={trace_id}] 已调用MCP工具 {tool_name}，结果={tool_result}")
                    system_prompt = (
                        "你是跨境电商智能运营助手。用户刚才问了天气/时间，"
                        "以下是实时工具查询结果，请自然、简洁地回复用户（不要输出任何内部标记、子任务信息）：\n"
                        f"{tool_result}"
                    )
                else:
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

    # ==========【复合意图统一规划：sub_results非空时，按 group_id 分组，同组子意图结果一起做衔接推理】==========
    if agent.sub_executed and agent.sub_results:
        print(f"[result_collect_node][trace_id={trace_id}] 复合意图统一规划，共 {len(agent.sub_results)} 个子结果，开始按组聚合")
        # 1. 按 group_id 分组（保持原始出现顺序），同属一个上层问题的子意图放同一组
        grouped = {}
        group_order = []
        for sr in agent.sub_results:
            gid = sr.get("group_id", 0)
            if gid not in grouped:
                grouped[gid] = []
                group_order.append(gid)
            grouped[gid].append(sr)

        # 2. 每组内部组织成块（保留类型和原文，方便LLM理解组内衔接），组间按顺序拼接
        group_blocks = []
        for gid in group_order:
            members = grouped[gid]
            member_lines = []
            for idx, sr in enumerate(members, start=1):
                member_lines.append(
                    f"  子项{idx}·类型:{sr.get('type')}｜原文:{sr.get('intent', '')}\n  结果:{sr.get('result', '')}"
                )
            # 组内多个子意图说明是同一上层问题的不同来源，需联合推理
            group_note = (
                "（组内多个子项属于同一个上层问题，需将它们的结果联合起来做衔接推理）"
                if len(members) > 1 else ""
            )
            group_blocks.append(
                f"【问题组{gid}】{group_note}\n" + "\n".join(member_lines)
            )
        combined_material = "\n\n".join(group_blocks)

        from utils.skill_loader import get_skill_loader
        # 如果输出反思发现了问题，重新整合时把问题反馈给大模型
        feedback_line = ""
        if agent.output_reflection_issues:
            feedback_line = f"\n\n⚠️上次回答被质检员指出以下问题，请这次重点修正和补全：\n{agent.output_reflection_issues}"

        plan_prompt = f"""用户提出了包含多个方面的问题，Agent 已分别处理，得到多个子结果（已按问题分组）：
{combined_material}
{feedback_line}

请把这些结果按逻辑顺序组织成一个连贯、完整、面向用户的回答。
要求：
1. 覆盖所有分组，不要遗漏任何一组；
2. 同一"问题组"内如果有因果/衔接关系（例如"天气是否影响物流"，需结合该组内的天气数据和物流规则），必须在回答中把组内内容衔接起来讲清楚；
3. 不同问题组之间，按用户提问的自然顺序依次回答；
4. 语气自然、口语化，像真人客服，不暴露任何内部标记、分组编号、子任务信息。"""
        try:
            result_collect_rule = get_skill_loader().load_rule("result_collect")
            system_content = result_collect_rule if result_collect_rule else "你是跨境电商智能运营助手，负责把多个回答片段整理成连贯的最终回答。"
            llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
            resp = llm.invoke([
                {"role": "system", "content": system_content},
                {"role": "user", "content": plan_prompt}
            ])
            final_answer = resp.content.strip()
            print(f"[result_collect_node][trace_id={trace_id}] 复合意图分组衔接推理结果：\n{final_answer}")
        except Exception as e:
            print(f"[result_collect_node][trace_id={trace_id}] 复合意图统一规划异常: {str(e)}")
            # 兜底：把子结果按组简单拼接，防止链路崩溃
            joined = "\n".join(sr.get('result', '') for sr in agent.sub_results)
            final_answer = joined

        updated_state_dict = update_agent_state(state, final_answer=final_answer)
        new_ai_msg = AIMessage(content=final_answer)
        # 只返回新增消息，让MessagesState的add_messages reducer自动追加/替换
        if agent.output_reflection_retry > 0 and state["messages"]:
            # 反思重试：给新消息设相同id，让reducer替换最后一条旧AI消息
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "id") and last_msg.id:
                new_ai_msg.id = last_msg.id
        updated_state_dict["messages"] = [new_ai_msg]
        return updated_state_dict

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
            final_answer = resp.content.strip()
            print(f"[result_collect_node][trace_id={trace_id}] LLM整理后的结果：\n{final_answer}")
        except Exception as e:
            print(f"[result_collect_node][trace_id={trace_id}] LLM文本整理异常: {str(e)}")
            # LLM调用失败或规则缺失兜底，直接原始素材输出，防止链路崩溃
            final_answer = str(raw_source_data)

    # 更新agent_state内部字段final_answer
    updated_state_dict = update_agent_state(
        state,
        final_answer=final_answer
    )
    # 只返回新增AIMessage，让add_messages reducer自动追加，不要返回完整列表
    new_ai_msg = AIMessage(content=final_answer)
    updated_state_dict["messages"] = [new_ai_msg]
    return updated_state_dict


def output_reflection_node(state: GraphState) -> dict[str, Any]:
    """
    【输出反思节点】校验最终回答是否真正覆盖了用户的原始问题
    输入：用户原始问题 + 切分的子意图列表 + 大模型刚整合好的回答
    输出：通过则结束；不通过则记录问题，路由回 result_collect 重新整合（最多重试1次）
    """
    agent = state["agent_state"]
    trace_id = agent.trace_id
    user_query = agent.user_query
    final_answer = agent.final_answer
    intent_list = agent.intent_list

    print(f"[output_reflection_node][trace_id={trace_id}] 开始输出反思校验")

    # 构造子意图清单，方便大模型对照检查
    intent_summary = "\n".join([
        f"  子项{i+1}: {it.get('intent', '')} (类型:{it.get('type', '')}, business:{it.get('business', '-')})"
        for i, it in enumerate(intent_list)
    ])

    # 构造各子意图实际执行结果，让反思节点知道哪些信息是真实查到的
    sub_results_summary = ""
    try:
        sr_list = agent.sub_results or []
        if sr_list:
            sr_lines = []
            for i, sr in enumerate(sr_list):
                r_type = sr.get("type", "")
                r_intent = sr.get("intent", "")
                r_result = str(sr.get("result", ""))[:300]
                sr_lines.append(f"  [{r_type}] {r_intent} → {r_result}")
            sub_results_summary = "各子意图实际执行结果（这些是真实查到的信息，不算编造）：\n" + "\n".join(sr_lines) + "\n"
    except Exception:
        pass

    # 取最近几轮对话历史，让反思节点知道用户之前说过什么（否则会把基于历史的回答误判为编造）
    recent_history = ""
    try:
        msgs = state.get("messages", [])
        # 取最近6条（3轮对话），去掉最后一条刚加的HumanMessage
        recent = msgs[-7:-1] if len(msgs) > 1 else []
        if recent:
            history_lines = []
            for m in recent:
                role = "用户" if m.type == "human" else "助手"
                history_lines.append(f"  [{role}] {m.content[:150]}")
            recent_history = "对话历史（用户之前说过的内容）：\n" + "\n".join(history_lines) + "\n"
    except Exception:
        pass

    prompt = f"""你是回答质检员。请对照用户原始问题和最终回答，检查回答是否合格。

{recent_history}
{sub_results_summary}
用户当前问题：{user_query}

系统拆出的子意图清单：
{intent_summary}

系统给出的最终回答：
{final_answer}

请严格按以下标准检查，只输出JSON：
{{"pass": true/false, "issues": "如果不通过，说明具体遗漏了什么或哪里答偏了；如果通过就留空"}}

判断标准：
1. 用户问的每个子意图，回答里都要有所体现，不能遗漏；
2. 回答不能答非所问；如果信息来自对话历史或上面"各子意图实际执行结果"，不算编造；
3. 如果某个子意图执行结果是"没有找到/查不到"，回答里要诚实说明，不能把别的子意图结果张冠李戴；
4. 只输出JSON，不要多余文字。"""

    passed = False
    issues = ""
    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        resp = llm.invoke(prompt)
        data = safe_parse_json(resp.content.strip(), trace_id)
        passed = bool(data.get("pass", False))
        issues = str(data.get("issues", ""))
    except Exception as e:
        print(f"[output_reflection_node][trace_id={trace_id}] 输出反思解析异常: {str(e)}")
        # 反思本身出错，不阻断流程，直接放行
        passed = True
        issues = ""

    if passed:
        print(f"[output_reflection_node][trace_id={trace_id}] 输出反思通过，回答合格")
        return update_agent_state(state, output_reflection_issues="")
    else:
        retry = agent.output_reflection_retry
        print(f"[output_reflection_node][trace_id={trace_id}] 输出反思发现问题(第{retry+1}次): {issues[:100]}")
        # 如果已经重试过1次还是不通过，在最终回答前面加提示，然后结束
        if retry >= 1:
            warning = "⚠️ 以上回答可能存在遗漏或考虑不完整，如需更准确的信息请补充说明后再问。\n\n"
            warned_answer = warning + agent.final_answer
            print(f"[output_reflection_node][trace_id={trace_id}] 已达重试上限，附加不完整提示后结束")
            result = update_agent_state(
                state,
                final_answer=warned_answer,
                output_reflection_issues=""  # 清空，让路由走END
            )
            # 给新AIMessage设相同id，让reducer替换最后一条旧AI消息
            if state["messages"]:
                last_msg = state["messages"][-1]
                warned_msg = AIMessage(content=warned_answer)
                if hasattr(last_msg, "id") and last_msg.id:
                    warned_msg.id = last_msg.id
                result["messages"] = [warned_msg]
            return result
        else:
            return update_agent_state(
                state,
                output_reflection_issues=issues,
                output_reflection_retry=retry + 1
            )


def route_after_output_reflection(state: GraphState) -> Literal["result_collect", "__end__"]:
    """
    输出反思后的路由：
    - 已重试过1次 → 直接结束（避免死循环）
    - 未通过且未超次数 → 回 result_collect 重新整合
    - 通过 → 直接结束
    """
    agent = state["agent_state"]
    retry = agent.output_reflection_retry
    issues = agent.output_reflection_issues

    if issues and retry <= 1:
        print(f"[route_after_output_reflection][trace_id={agent.trace_id}] 回答有问题，回result_collect重新整合(第{retry}次)")
        return "result_collect"
    else:
        print(f"[route_after_output_reflection][trace_id={agent.trace_id}] 回答合格或已达重试上限，结束")
        return "__end__"


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
