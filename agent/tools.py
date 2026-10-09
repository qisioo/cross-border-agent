""" agent/tools.py 业务工具集合，普通函数封装（规避BaseTool与PydanticV2版本冲突）
职责：所有外部能力封装，节点(nodes/graph_builder)只调用工具，不写业务实现
包含：商品信息抓取、竞品分析、listing生成、销售数据查询、客服RAG问答
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import hashlib
import requests
from bs4 import BeautifulSoup
import time
import os
import datetime

# agent/tools.py 顶部增加
from agent.output_parsers import ListingOutputSchema as ListingOutput



from langchain_deepseek import ChatDeepSeek
from agent.output_parsers import parse_listing_output, parse_sales_params

logger = logging.getLogger(__name__)

# --------------------------
# 输入参数Schema定义（用于参数校验）
# --------------------------
class ProductFetchInput(BaseModel):
    product_keyword: str = Field(description="商品关键词/商品链接")
    trace_id: str = Field(description="链路追踪ID")

class CompetitorAnalyzeInput(BaseModel):
    product_name: str = Field(description="目标商品名称")
    price_range: Optional[str] = Field(default="", description="价格区间")
    trace_id: str = Field(description="链路追踪ID")

class ListingGenInput(BaseModel):
    product_info: Dict[str, Any] = Field(description="商品基础信息字典")
    target_market: str = Field(description="目标站点市场")
    trace_id: str = Field(description="链路追踪ID")

class SalesQueryInput(BaseModel):
    user_query: str = Field(description="用户原始自然语言查询")
    trace_id: str = Field(description="链路追踪ID")

class CustomerServiceInput(BaseModel):
    question: str = Field(description="用户客服提问")
    trace_id: str = Field(description="链路追踪ID")

# --------------------------
# 公共工具函数
# --------------------------
def calc_doc_hash(doc) -> str:
    """文档片段哈希，和hybrid_retriever保持一致，用于来源标记"""
    content = doc.page_content.strip()
    return hashlib.md5(content.encode("utf-8")).hexdigest()

# --------------------------
# 工具函数实现
# --------------------------
def product_info_fetch(product_keyword: str, trace_id: str) -> Dict[str, Any]:
    """
    根据商品关键词或者链接，抓取商品基础信息：标题、卖点、参数、图片等
    自动判断：如果product_keyword是http/https链接，则执行网页爬虫；
    非URL输入不再提供mock模拟数据，直接返回错误。
    """
    try:
        # 参数校验
        input_model = ProductFetchInput(product_keyword=product_keyword, trace_id=trace_id)
        logger.debug(f"[product_info_fetch][trace_id={input_model.trace_id}] 开始抓取商品信息 keyword={input_model.product_keyword}")

        # 判断输入是否为url链接
        url_prefix = ("http://", "https://")
        if input_model.product_keyword.startswith(url_prefix):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            # 同步休眠，防高频请求，同步模式可用；异步场景需替换为asyncio.sleep
            time.sleep(0.5)
            resp = requests.get(input_model.product_keyword, headers=headers, timeout=10)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding  # 新增：自动识别网页编码，解决部分页面中文乱码
            soup = BeautifulSoup(resp.text, "html.parser")

            # 提取页面标题
            page_title = None
            title_tag = soup.find("title")
            if title_tag:
                page_title = title_tag.get_text(strip=True)
            h1_tag = soup.find("h1")
            if h1_tag and not page_title:
                page_title = h1_tag.get_text(strip=True)

            # ===== 适配 books.toscrape 页面，提取价格、商品卖点描述 =====
            price = None
            selling_points = []
            # 价格选择器
            price_tag = soup.select_one("p.price_color")
            if price_tag:
                price = price_tag.get_text(strip=True)
            # 商品描述
            desc_tag = soup.select_one("#product_description + p")
            if desc_tag:
                selling_points.append(desc_tag.get_text(strip=True))

            logger.debug(f"[product_info_fetch][trace_id={input_model.trace_id}] 抓取解析结果 title={page_title}, price={price}")
            # 构造返回结构，schema保持不变
            result_data = {
                "title": page_title or "未知商品",
                "brand": "DemoBookStore",
                "spec": None,
                "selling_points": selling_points,
                "price": price,
                "source_url": input_model.product_keyword
            }
            logger.debug(f"[product_info_fetch][trace_id={input_model.trace_id}] 网页抓取完成")
            return result_data
        else:
            # 【移除全部mock，非URL直接报错】
            err_msg = "product_info_fetch仅支持传入网页URL进行抓取；关键词抓取逻辑已移除，请输入商品详情链接"
            logger.warning(f"[product_info_fetch][trace_id={input_model.trace_id}] {err_msg}")
            return {"error": err_msg}

    except requests.exceptions.RequestException as e:
        err_msg = f"网页请求异常: {str(e)}"
        logger.error(f"[product_info_fetch][trace_id={trace_id}] {err_msg}")
        return {"error": err_msg}
    except Exception as e:
        logger.error(f"[product_info_fetch][trace_id={trace_id}] 工具执行异常: {str(e)}")
        return {"error": str(e)}


logger = logging.getLogger(__name__)

def fetch_product_list(list_page_url: str, trace_id: str, limit: int = 3) -> Dict[str, Any]:
    """
    批量抓取商品列表页，提取多个商品详情链接，调用product_info_fetch获取商品信息
    :param list_page_url: 商品列表页面url
    :param trace_id: 链路追踪ID
    :param limit: 最多抓取多少个商品，防止抓取过多
    :return: 包含商品列表的字典
    """
    try:
        # 参数校验
        class ProductListInput(BaseModel):
            list_page_url: str = Field(description="商品列表页面链接")
            trace_id: str = Field(description="链路追踪ID")
            limit: int = Field(default=3, description="最大抓取商品数量")
        input_model = ProductListInput(list_page_url=list_page_url, trace_id=trace_id, limit=limit)
        logger.debug(f"[fetch_product_list][trace_id={input_model.trace_id}] 开始解析商品列表页 url={input_model.list_page_url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(input_model.list_page_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # books.toscrape 列表页的商品a标签选择器
        links = soup.select("article.product_pod h3 a")
        detail_urls = []
        for a_tag in links[:input_model.limit]:
            relative_path = a_tag["href"]
            # 拼接相对路径为完整url
            if relative_path.startswith(("http://", "https://")):
                full_url = relative_path
            elif relative_path.startswith("catalogue/"):
                full_url = f"https://books.toscrape.com/{relative_path}"
            else:
                full_url = f"https://books.toscrape.com/catalogue/{relative_path}"

            detail_urls.append(full_url)

        logger.debug(f"[fetch_product_list][trace_id={input_model.trace_id}] 解析到{len(detail_urls)}个商品链接，开始批量抓取")
        product_list = []
        batch_start_time = time.time()
        BATCH_MAX_DURATION = 35
        is_batch_timeout = False  # 新增：超时标记

        for url in detail_urls:
            # 总耗时判断：超过阈值标记超时，终止剩余抓取
            if time.time() - batch_start_time > BATCH_MAX_DURATION:
                logger.warning(f"[fetch_product_list][trace_id={input_model.trace_id}] 整批抓取超过{BATCH_MAX_DURATION}s，停止抓取剩余商品")
                is_batch_timeout = True
                break

            logger.debug(f"[fetch_product_list][trace_id={input_model.trace_id}] 开始抓取详情页 {url}")
            item_data = product_info_fetch(product_keyword=url, trace_id=input_model.trace_id)
            if "error" not in item_data:
                product_list.append(item_data)
                logger.debug(f"[fetch_product_list][trace_id={input_model.trace_id}] 详情页抓取成功 {url}")
            else:
                logger.warning(f"[fetch_product_list][trace_id={input_model.trace_id}] 详情页抓取失败 {url}, msg={item_data['error']}")
            time.sleep(0.6)  # 每个商品之间增加延时，防请求过快

        # 组装返回结果，新增提示字段
        result = {
            "list_page_url": input_model.list_page_url,
            "count": len(product_list),
            "product_items": product_list,
        }
        # 超时标记：增加提示信息
        if is_batch_timeout:
            result["warning_msg"] = "商品未能全部抓取：批量抓取超时，部分商品未采集"

        return result

    except Exception as e:
        logger.error(f"[fetch_product_list][trace_id={trace_id}] 批量抓取异常: {str(e)}")
        return {"error": str(e)}

# ========== 新增：多页循环抓取函数 ==========
def fetch_product_list_pages(base_url: str, trace_id: str, max_page: int = 2) -> Dict[str, Any]:
    """
    多页循环抓取商品列表，复用fetch_product_list单页抓取逻辑
    :param base_url: 列表基础url模板，例如 https://books.toscrape.com/catalogue/page-{}.html
    :param trace_id: 链路追踪ID
    :param max_page: 最大抓取页数，循环条件：页码 < max_page。max_page=2 只抓取第1页
    :return: 汇总后的全部商品结果
    """
    try:
        logger.debug(f"[fetch_product_list_pages][trace_id={trace_id}] 开始多页抓取，max_page={max_page}")
        all_products = []
        """
        :param max_page: 最大抓取页数，循环范围 [1, max_page‑1]。max_page=2 →仅抓取第1页
        """
        for page_num in range(1, max_page):
            page_url = base_url.format(page_num)
            logger.debug(f"[fetch_product_list_pages][trace_id={trace_id}] 抓取页码：{page_num}, url={page_url}")
            batch_res = fetch_product_list(list_page_url=page_url, trace_id=trace_id, limit=20)
            if "error" in batch_res:
                logger.error(f"[fetch_product_list_pages][trace_id={trace_id}] 页码{page_num}抓取失败: {batch_res['error']}")
                continue
            items = batch_res.get("product_items", [])
            if not items:
                logger.info(f"[fetch_product_list_pages][trace_id={trace_id}] 页码{page_num}无商品，停止抓取")
                break
            all_products.extend(items)
            time.sleep(0.8)
        return {
            "base_url": base_url,
            "max_page": max_page,
            "total_count": len(all_products),
            "product_items": all_products
        }
    except Exception as e:
        logger.error(f"[fetch_product_list_pages][trace_id={trace_id}] 多页抓取异常: {str(e)}")
        return {"error": str(e)}

def competitor_analysis(product_name: str, trace_id: str, price_range: str = "") -> str:
    """
    竞品分析工具：调用fetch_product_list抓取公开竞品页面，基于真实抓取内容生成报告，移除纯硬编码mock
    """
    try:
        input_model = CompetitorAnalyzeInput(product_name=product_name, price_range=price_range, trace_id=trace_id)
        logger.debug(f"[competitor_analysis][trace_id={input_model.trace_id}] 竞品分析开始 product={input_model.product_name}, price_range={input_model.price_range}")

        # 使用公开图书站点作为竞品数据源
        list_url = "https://books.toscrape.com/catalogue/page-1.html"
        batch_result = fetch_product_list(list_page_url=list_url, trace_id=trace_id, limit=3)
        if "error" in batch_result:
            raise Exception(f"竞品列表抓取失败: {batch_result['error']}")

        product_items = batch_result.get("product_items", [])
        competitor_text = ""
        for idx, item in enumerate(product_items):
            competitor_text += f"""竞品{idx+1}: 标题: {item.get('title')} 价格: {item.get('price')} 卖点: {item.get('selling_points')} 来源: {item.get('source_url')} """
        prompt = f"""
你是跨境电商竞品分析师，基于下面抓取到的竞品原始页面信息，生成竞品分析报告。
目标商品名称：{product_name}，价格区间：{price_range if price_range else "不限"}
竞品原始数据：
{competitor_text}
输出报告，包含竞品基础信息、优劣势、市场机会。
"""
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        report = llm.invoke(prompt).content
        logger.debug(f"[competitor_analysis][trace_id={input_model.trace_id}] 竞品分析完成")
        return report
    except Exception as e:
        logger.error(f"[competitor_analysis][trace_id={trace_id}] 工具执行异常: {str(e)}")
        return f"【竞品分析失败】{str(e)}"


def competitor_analysis_parallel(task_name: str, product_items: list, trace_id: str, warning_msg: str = "") -> str:
    """
    竞品分析子任务执行器：基于已抓取的竞品原始数据，按子任务角度生成分析片段
    配合exec_task_dispatcher并行调度使用，避免每个子任务重复爬虫触发站点限流
    :param task_name: 子任务名称（如：收集竞品基础信息）
    :param product_items: 已抓取的竞品商品列表（fetch_product_list返回值product_items）
    :param trace_id: 链路追踪ID
    :param warning_msg: 数据采集警告信息，为空代表无警告
    :return: 该子任务的分析文本
    """
    try:
        competitor_text = ""
        for idx, item in enumerate(product_items):
            competitor_text += f"""竞品{idx + 1}: 标题: {item.get('title')} 价格: {item.get('price')} 卖点: {item.get('selling_points')} 来源: {item.get('source_url')} """

        prompt_prefix = ""
        if warning_msg:
            prompt_prefix = f"【数据采集备注】{warning_msg}\n> 注意：本次分析仅基于已成功抓取到的商品数据，未抓取的商品不纳入分析范围。\n\n"

        prompt = f"""{prompt_prefix}你是跨境电商竞品分析师，基于下面抓取到的竞品原始页面信息，完成指定子任务分析。
子任务：{task_name}
竞品原始数据：
{competitor_text}
请围绕该子任务输出对应的分析内容，简洁精炼，只输出该子任务的分析结果，不要输出子任务标题以外的多余内容。
"""
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        report = llm.invoke(prompt).content
        logger.debug(f"[competitor_analysis_parallel][trace_id={trace_id}] 子任务完成: {task_name}")
        return report
    except Exception as e:
        logger.error(f"[competitor_analysis_parallel][trace_id={trace_id}] 子任务执行异常: {str(e)}")
        return f"【子任务失败】{str(e)}"


from langchain_core.output_parsers import PydanticOutputParser


import json
def listing_generate(product_info: Dict[str, Any], target_market: str, trace_id: str) -> Dict[str, str]:
    """
    根据商品信息生成跨境电商Listing：标题、五点描述、广告文案
    ✅ Prompt + PydanticOutputParser双重格式校验，严格英文输出，字段强校验，带重试机制
    """
    try:
        input_model = ListingGenInput(product_info=product_info, target_market=target_market, trace_id=trace_id)
        logger.debug(f"[listing_generate][trace_id={input_model.trace_id}] Listing生成 market={input_model.target_market}")

        # 【核心改动】使用Pydantic模型自动生成格式要求，不再手写JSON模板
        parser = PydanticOutputParser(pydantic_object=ListingOutput)
        format_instruct = parser.get_format_instructions()

        # 序列化商品信息，防止字典直接转字符串格式混乱
        product_json_str = json.dumps(product_info, ensure_ascii=False, indent=2)

        # ==========【Skill规则注入】B类固定规则下沉到skill/rules/listing.md，Skill结果优先，失败抛错走上层重试 ==========
        from utils.skill_loader import get_skill_loader
        listing_rule = get_skill_loader().load_rule("listing")
        if not listing_rule:
            raise Exception("Listing业务规则加载失败（skill/rules/listing.md缺失），无法生成")
        # ======================================================================

        base_prompt = f"""你是亚马逊跨境Listing文案专家，根据商品信息生成英文Listing。
请严格遵守以下Listing业务规则：
{listing_rule}

{format_instruct}

商品信息：
{product_json_str}
目标市场：{target_market}
"""
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        max_retry = 2
        validated_result = None
        current_prompt = base_prompt
        actual_retry_count = 0  # 新增：统计真实发生的重试次数

        for retry in range(max_retry):
            raw_resp = llm.invoke(current_prompt).content
            validated_result = parse_listing_output(raw_resp, trace_id)
            if validated_result is not None:
                break
            actual_retry_count += 1
            logger.warning(
                f"[listing_generate][trace_id={trace_id}] Listing格式校验失败，第{retry+1}次重试生成，raw_output={raw_resp[:200]}"
            )
            # 重试追加提示，纠正LLM（固定校验规则已由Skill注入，此处仅保留任务指令）
            current_prompt = base_prompt + "\n上一轮输出格式校验失败，请重新生成并严格遵守上述业务规则与输出格式要求。"

        logger.info(
            f"[listing_generate][trace_id={trace_id}] Listing生成结束，实际重试次数={actual_retry_count}, max_retry={max_retry}"
        )

        if validated_result is None:
            logger.error(f"[listing_generate][trace_id={trace_id}] Listing输出校验失败，已用尽全部重试次数")
            raise Exception(f"Listing输出校验失败，已重试{max_retry}次，不符合格式规范")

        return {
            "title": validated_result.title,
            "bullet_points": "\n".join(validated_result.bullet_points),
            "ad_copy": validated_result.ad_copy
        }
    except Exception as e:
        logger.error(f"[listing_generate][trace_id={trace_id}] 工具执行异常: {str(e)}")
        return {"error": str(e)}


def sales_data_query(user_query: str, trace_id: str) -> Dict[str, Any]:
    """
    ✅ 销售数据查询：自然语言提取商品ID、月份，读取本地CSV销售文件
    :param user_query: 用户原始自然语言，例如：查询SKU001在2026-09的销售数据
    """
    try:
        import csv
        import re
        import json
        input_model = SalesQueryInput(user_query=user_query, trace_id=trace_id)
        logger.debug(f"[sales_data_query][trace_id={input_model.trace_id}] 销售查询，原始query={input_model.user_query}")

        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        max_retry = 2
        params = None
        # ==========【Skill规则注入】B类字段规范下沉到skill/rules/sales.md，Skill结果优先，失败抛错走上层重试 ==========
        from utils.skill_loader import get_skill_loader
        sales_rule = get_skill_loader().load_rule("sales")
        if not sales_rule:
            raise Exception("销售查询业务规则加载失败（skill/rules/sales.md缺失），无法执行")
        # ======================================================================

        # 直接拼接避免.format与规则内JSON示例花括号冲突
        base_prompt = (
            "从用户提问提取商品ID(product_id)、月份(month)。\n"
            "请严格遵守以下销售查询业务规则：\n"
            f"{sales_rule}\n\n"
            f"用户问题：{user_query}\n"
        )
        current_prompt = base_prompt

        for retry in range(max_retry):
            raw_resp = llm.invoke(current_prompt).content
            clean_text = raw_resp.strip()
            # 移除markdown代码块
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            # 多层JSON解析兜底
            parsed_dict = None
            try:
                parsed_dict = json.loads(clean_text)
            except Exception:
                # 处理外层包裹双引号的情况
                if clean_text.startswith('"') and clean_text.endswith('"'):
                    try:
                        parsed_dict = json.loads(json.loads(clean_text))
                    except Exception:
                        parsed_dict = None

            pid_val = None
            mon_val = None
            if isinstance(parsed_dict, dict):
                pid_val = str(parsed_dict.get("product_id", "")).strip()
                mon_val = str(parsed_dict.get("month", "")).strip()

            # 字典解析失败，正则兜底
            if not pid_val or not mon_val:
                pid_match = re.search(r'"product_id"\s*:\s*"([^"]+)"', clean_text)
                mon_match = re.search(r'"month"\s*:\s*"(\d{4}-\d{2})"', clean_text)
                pid_val = pid_match.group(1).strip() if pid_match else None
                mon_val = mon_match.group(1).strip() if mon_match else None

            if pid_val and mon_val:
                from agent.output_parsers import SalesQueryParamSchema
                params = SalesQueryParamSchema(product_id=pid_val, month=mon_val)
                break

            logger.warning(f"[sales_data_query][trace_id={trace_id}] 参数提取失败，第{retry + 1}次重试，LLM原始输出前200字：{clean_text[:200]}")
            current_prompt = base_prompt + "\n上一轮参数提取失败，请严格输出裸JSON，不要加引号、不要markdown。"

        if params is None or params.product_id is None or params.month is None:
            raise Exception("无法从用户语句解析出商品ID或者月份，重试耗尽")

        # ========== CSV读取 ==========
        csv_path = r"./rag/sales_data.csv"
        found_record = None
        try:
            # utf-8-sig 自动去除BOM头
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_pid = str(row.get("product_id", "")).strip()
                    row_month = str(row.get("month", "")).strip()
                    if row_pid == params.product_id and row_month == params.month:
                        found_record = {
                            "product_id": row_pid,
                            "month": row_month,
                            "sales_volume": int(row.get("sales_volume", 0)),
                            "revenue": int(row.get("revenue", 0)),
                            "conversion_rate": float(row.get("conversion_rate", 0))
                        }
                        break
        except UnicodeDecodeError:
            with open(csv_path, "r", encoding="gbk", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_pid = str(row.get("product_id", row.get("sku", ""))).strip()
                    row_month = str(row.get("month", row.get("月份", ""))).strip()
                    if row_pid == params.product_id and row_month == params.month:
                        found_record = {
                            "product_id": row_pid,
                            "month": row_month,
                            "sales_volume": int(row.get("sales_volume", 0)),
                            "revenue": int(row.get("revenue", 0)),
                            "conversion_rate": float(row.get("conversion_rate", 0))
                        }
                        break

        if found_record is None:
            raise Exception(f"不存在该SKU {params.product_id} 在{params.month}的销售记录")
        return found_record

    except FileNotFoundError:
        err_msg = "销售数据文件sales_data.csv不存在，请检查文件路径"
        logger.error(f"[sales_data_query][trace_id={trace_id}] {err_msg}")
        return {"error": err_msg}
    except Exception as e:
        logger.error(f"[sales_data_query][trace_id={trace_id}] 工具执行异常: {str(e)}")
        return {"error": str(e)}


def customer_service_qa(question: str, retrieve_docs: list, trace_id: str) -> Dict[str, Any]:
    """
    客服问答工具：基于RAG召回知识库片段，调用LLM生成客服回复
    :param question: 用户客服提问
    :param retrieve_docs: hybrid_retriever召回的文档列表（支持LangChain Document对象或序列化后的dict）
    :param trace_id: 链路追踪ID
    :return: 客服问答结果字典，包含回答、引用来源
    """
    try:
        input_model = CustomerServiceInput(question=question, trace_id=trace_id)
        logger.debug(
            f"[customer_service_qa][trace_id={input_model.trace_id}] 客服问答开始 question={input_model.question}")

        # 拼接知识库上下文，兼容两种类型：Document对象 / dict
        def get_doc_content(doc):
            if hasattr(doc, "page_content"):
                return doc.page_content
            elif isinstance(doc, dict) and "page_content" in doc:
                return doc["page_content"]
            else:
                return ""

        context_text = "\n".join([get_doc_content(doc) for doc in retrieve_docs])
        if not context_text.strip():
            answer = "抱歉，知识库中没有找到相关信息，请您更换问题再次咨询。"
        else:
            # ==========【Skill规则注入】给LLM的回答约束下沉到skill/rules/customer_service.md，Skill结果优先，失败抛错走外层兜底 ==========
            from utils.skill_loader import get_skill_loader
            cs_rule = get_skill_loader().load_rule("customer_service")
            if not cs_rule:
                raise Exception("知识库客服问答业务规则加载失败（skill/rules/customer_service.md缺失），无法执行")
            # ======================================================================
            prompt = f"""{cs_rule}

知识库参考内容：
{context_text}
用户问题：{question}
请基于上述知识库内容输出客服回复。
"""
            llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
            answer = llm.invoke(prompt).content.strip()

        source_list = []
        for idx, doc in enumerate(retrieve_docs):
            content = get_doc_content(doc)
            source_list.append({
                "doc_id": calc_doc_hash(doc),
                "page_content": content[:100]
            })
        result = {
            "answer": answer,
            "source": source_list
        }
        logger.debug(f"[customer_service_qa][trace_id={input_model.trace_id}] 客服问答完成")
        return result
    except Exception as e:
        logger.error(f"[customer_service_qa][trace_id={trace_id}] 客服问答工具异常: {str(e)}")
        return {"error": str(e)}


def save_product_to_knowledge(product_data: Dict[str, Any], save_dir: str, trace_id: str) -> Dict[str, Any]:
    """
    将抓取的商品信息保存为txt文件，放到rag/knowledge目录，用于后续向量入库
    :param product_data: product_info_fetch抓取返回的商品字典
    :param save_dir: 保存目录 D:\qi_agent\Agent03\rag\knowledge
    :param trace_id: 链路追踪ID
    :return: 文件保存结果
    """
    try:
        logger.debug(f"[save_product_to_knowledge][trace_id={trace_id}] 开始写入商品文档到 {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        # 构造文件名，使用trace_id防止重名
        file_name = f"product_{trace_id}.txt"
        file_path = os.path.join(save_dir, file_name)

        # 组装文本内容
        lines = []
        lines.append(f"商品标题：{product_data.get('title', '')}")
        lines.append(f"品牌：{product_data.get('brand', '')}")
        lines.append(f"规格：{product_data.get('spec', '')}")
        lines.append(f"价格：{product_data.get('price', '')}")
        lines.append(f"商品卖点：")
        for point in product_data.get("selling_points", []):
            lines.append(f"- {point}")
        lines.append(f"来源链接：{product_data.get('source_url', '')}")
        content = "\n".join(lines)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"[save_product_to_knowledge][trace_id={trace_id}] 文件保存成功，path={file_path}")
        return {"success": True, "file_path": file_path}
    except Exception as e:
        logger.error(f"[save_product_to_knowledge][trace_id={trace_id}] 保存文件失败: {str(e)}")
        return {"success": False, "error": str(e)}

def sync_list_page_to_single_know_file(list_page_url: str, save_dir: str, trace_id: str, limit: int = 3) -> Dict[str, Any]:
    """
    抓取列表页全部商品，把所有商品写入同一个txt文件
    页面无变化：跳过写入；页面商品变更：覆盖该txt文件
    :param list_page_url: 商品列表页面url
    :param save_dir: 知识库保存目录 rag/knowledge
    :param trace_id: 链路追踪ID
    :param limit: 最大抓取商品数量
    :return: 执行结果
    """
    try:
        logger.debug(f"[sync_list_page_to_single_know_file][trace_id={trace_id}] 开始同步列表页面 {list_page_url}")
        # 抓取列表页所有商品
        batch_result = fetch_product_list(list_page_url=list_page_url, trace_id=trace_id, limit=limit)
        if "error" in batch_result:
            return {"success": False, "error": f"列表抓取失败: {batch_result['error']}"}
        product_items = batch_result.get("product_items", [])
        if not product_items:
            logger.info(f"[sync_list_page_to_single_know_file][trace_id={trace_id}] 本次抓取无商品，不生成文件")
            return {"success": True, "msg": "无商品数据，跳过写入"}

        os.makedirs(save_dir, exist_ok=True)
        # 用url的md5作为文件名，同一个列表页固定同一个文件
        url_hash = hashlib.md5(list_page_url.encode("utf-8")).hexdigest()
        file_name = f"product_list_{url_hash}.txt"
        file_path = os.path.join(save_dir, file_name)

        # 组装全部商品文本内容
        content_lines = []
        content_lines.append(f"【来源页面】{list_page_url}")
        content_lines.append(f"【抓取时间】{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_lines.append(f"【本次抓取商品总数】{len(product_items)}")
        content_lines.append("=" * 70)

        for idx, prod in enumerate(product_items, start=1):
            content_lines.append(f"==== 商品 {idx} ====")
            content_lines.append(f"标题：{prod.get('title','')}")
            content_lines.append(f"品牌：{prod.get('brand','')}")
            content_lines.append(f"规格：{prod.get('spec','')}")
            content_lines.append(f"价格：{prod.get('price','')}")
            content_lines.append(f"来源链接：{prod.get('source_url','')}")
            content_lines.append("卖点：")
            for point in prod.get("selling_points", []):
                content_lines.append(f"- {point}")
            content_lines.append("\n")
        full_text = "\n".join(content_lines)

        # 计算当前这批商品的指纹，用于对比是否发生变化
        raw_finger_str = ""
        for prod in product_items:
            item_str = f"{prod.get('source_url','')}|{prod.get('title','')}|{prod.get('price','')}|{prod.get('selling_points','')}"
            raw_finger_str += item_str
        current_finger = hashlib.md5(raw_finger_str.encode("utf-8")).hexdigest()

        # 如果文件已存在，读取旧文件，对比指纹
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_content = f.read()
            # 修复：直接对磁盘旧文件内容做md5指纹，不需要解析商品
            old_finger = hashlib.md5(old_content.encode("utf-8")).hexdigest()
            if current_finger == old_finger:
                logger.info(f"[sync_list_page_to_single_know_file][trace_id={trace_id}] 页面商品数据无变化，跳过写入文件 {file_name}")
                return {"success": True, "msg": "页面数据未变更，跳过写入", "file_path": file_path}

        # 数据发生变化，覆盖写入完整新内容
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"[sync_list_page_to_single_know_file][trace_id={trace_id}] 已写入{len(product_items)}个商品到 {file_path}")
        return {"success": True, "msg": "写入成功", "file_path": file_path, "product_count": len(product_items)}

    except Exception as e:
        logger.error(f"[sync_list_page_to_single_know_file][trace_id={trace_id}] 同步文件异常: {str(e)}")
        return {"success": False, "error": str(e)}

# ========== 新增：多页抓取，全部商品写入同一个txt ==========
def sync_multi_page_to_single_know_file(base_url: str, save_dir: str, trace_id: str, max_page: int = 2) -> Dict[str, Any]:
    """
    多页抓取全部商品，合并写入同一个知识库txt文件
    页面无变化：跳过写入；商品发生变更：覆盖文件
    """
    try:
        logger.debug(f"[sync_multi_page_to_single_know_file][trace_id={trace_id}] 多页同步开始 max_page={max_page}")
        batch_result = fetch_product_list_pages(base_url=base_url, trace_id=trace_id, max_page=max_page)
        if "error" in batch_result:
            return {"success": False, "error": f"多页抓取失败: {batch_result['error']}"}
        product_items = batch_result.get("product_items", [])
        if not product_items:
            logger.info(f"[sync_multi_page_to_single_know_file][trace_id={trace_id}] 本次抓取无商品，跳过写入")
            return {"success": True, "msg": "无商品数据，跳过写入"}

        os.makedirs(save_dir, exist_ok=True)
        url_hash = hashlib.md5(base_url.encode("utf-8")).hexdigest()
        file_name = f"multi_page_product_list_{url_hash}.txt"
        file_path = os.path.join(save_dir, file_name)

        content_lines = []
        content_lines.append(f"【多页来源模板】{base_url}")
        content_lines.append(f"【最大抓取页数设置】{max_page}")
        content_lines.append(f"【抓取时间】{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_lines.append(f"【本次抓取商品总数】{len(product_items)}")
        content_lines.append("=" * 70)

        for idx, prod in enumerate(product_items, start=1):
            content_lines.append(f"==== 商品 {idx} ====")
            content_lines.append(f"标题：{prod.get('title','')}")
            content_lines.append(f"品牌：{prod.get('brand','')}")
            content_lines.append(f"规格：{prod.get('spec','')}")
            content_lines.append(f"价格：{prod.get('price','')}")
            content_lines.append(f"来源链接：{prod.get('source_url','')}")
            content_lines.append("卖点：")
            for point in prod.get("selling_points", []):
                content_lines.append(f"- {point}")
            content_lines.append("\n")
        full_text = "\n".join(content_lines)

        # 计算全部商品指纹
        raw_finger_str = ""
        for prod in product_items:
            item_str = f"{prod.get('source_url','')}|{prod.get('title','')}|{prod.get('price','')}|{prod.get('selling_points','')}"
            raw_finger_str += item_str
        current_finger = hashlib.md5(raw_finger_str.encode("utf-8")).hexdigest()

        if os.path.exists(file_path):
            old_finger_str = ""
            for prod in product_items:
                item_str = f"{prod.get('source_url','')}|{prod.get('title','')}|{prod.get('price','')}|{prod.get('selling_points','')}"
                old_finger_str += item_str
            old_finger = hashlib.md5(old_finger_str.encode("utf-8")).hexdigest()
            if current_finger == old_finger:
                logger.info(f"[sync_multi_page_to_single_know_file][trace_id={trace_id}] 多页商品数据无变化，跳过写入 {file_name}")
                return {"success": True, "msg": "页面数据未变更，跳过写入", "file_path": file_path, "product_count": len(product_items)}

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"[sync_multi_page_to_single_know_file][trace_id={trace_id}] 多页同步完成，共{len(product_items)}个商品写入 {file_path}")
        return {"success": True, "msg": "写入成功", "file_path": file_path, "product_count": len(product_items)}

    except Exception as e:
        logger.error(f"[sync_multi_page_to_single_know_file][trace_id={trace_id}] 多页同步异常: {str(e)}")
        return {"success": False, "error": str(e)}

# 工具实例统一导出（函数）
__all__ = [
    "product_info_fetch",
    "fetch_product_list",
    "fetch_product_list_pages",
    "save_product_to_knowledge",
    "sync_list_page_to_single_know_file",
    "sync_multi_page_to_single_know_file",
    "competitor_analysis",
    "listing_generate",
    "sales_data_query",
    "customer_service_qa"
]

# ===================== 模块自测 =====================
if __name__ == "__main__":
    print("==== agent.tools 模块自测 ====")
    from utils.helper import generate_trace_id
    from langchain_core.documents import Document
    test_trace_id = generate_trace_id()

    # ========== 【默认执行：核心Mock工具测试（无爬虫，速度快）】 ==========
    # 测试商品抓取工具 - 关键词mock模式（不跑爬虫）
    print("\n【商品抓取工具 - 关键词Mock模式测试】")
    res2 = product_info_fetch(product_keyword="无线蓝牙耳机", trace_id=test_trace_id)
    print(res2)

    # 测试Listing生成（LLM+输出校验）
    res3 = listing_generate(product_info=res2, target_market="US", trace_id=test_trace_id)
    print("\n【Listing生成返回】")
    print(res3)

    # 测试销售查询：自然语言参数解析
    print("\n【销售数据查询（自然语言参数解析）】")
    res4 = sales_data_query(user_query="查询SKU001在2026-09的销售数据", trace_id=test_trace_id)
    print(res4)

    # 客服工具自测
    test_docs = [Document(page_content="蓝牙耳机：单次续航6小时，总续航30小时，IP54防尘防水")]
    res5 = customer_service_qa(question="耳机防水等级是多少？", retrieve_docs=test_docs, trace_id=test_trace_id)
    print("\n【客服问答工具返回】")
    print(res5["answer"])

    print("\n✅ tools【Mock核心自测】全部完成")

    # ========== 【可选爬虫测试块，默认注释，需要爬虫测试时取消注释】 ==========
    """
    # 测试竞品分析工具（真实爬虫抓取竞品）
    print("\n【竞品分析工具（爬虫）返回】")
    res1 = competitor_analysis(product_name="蓝牙耳机", price_range="20-50 USD", trace_id=test_trace_id)
    print(res1)

    # 测试商品抓取工具 - URL模式（爬虫）
    print("\n【商品抓取工具 - URL爬虫模式测试】")
    res2_url = product_info_fetch(product_keyword="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html", trace_id=test_trace_id)
    print(res2_url)

    # 批量抓取商品列表测试
    print("\n【批量商品列表抓取测试（爬虫）】")
    list_url = "https://books.toscrape.com/catalogue/page-1.html"
    batch_res = fetch_product_list(list_page_url=list_url, trace_id=test_trace_id, limit=20)
    print(batch_res)

    # 批量商品同步到单个知识库文件测试
    print("\n【批量商品同步到单个知识库文件测试（爬虫写入文件）】")
    sync_url = "https://books.toscrape.com/catalogue/page-1.html"
    sync_result = sync_list_page_to_single_know_file(
        list_page_url=sync_url,
        save_dir=r"D:\qi_agent\Agent03\rag\knowledge",
        trace_id=test_trace_id,
        limit=20
    )
    print(sync_result)
    print("\n✅ tools【爬虫扩展自测】全部完成")
    """


