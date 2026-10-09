""" 通用LLM输出解析工具 处理markdown代码块剥离、JSON清洗、解析容错 """
import json
import logging
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def strip_markdown_code_block(raw_text: str) -> str:
    """
    剥离 ```json ... ``` markdown标记
    """
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_parse_json(raw_text: str, trace_id: str) -> Optional[Dict[str, Any]]:
    """
    安全解析JSON：自动剥离代码块，解析失败返回None，打印日志
    :param raw_text: LLM原始返回字符串
    :param trace_id: 链路追踪ID
    :return: 解析后的字典，失败返回None
    """
    if not raw_text:
        logger.warning(f"[safe_parse_json][trace_id={trace_id}] 输入文本为空")
        return None
    clean_text = strip_markdown_code_block(raw_text)
    try:
        data = json.loads(clean_text)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[safe_parse_json][trace_id={trace_id}] JSON解析失败: {str(e)}, raw_clean_text={clean_text}")
        return None
    except Exception as e:
        logger.error(f"[safe_parse_json][trace_id={trace_id}] 未知解析异常: {str(e)}")
        return None


# ===================== Listing输出校验Schema =====================
class ListingOutputSchema(BaseModel):
    """跨境电商Listing输出结构，用于校验LLM返回结果"""
    title: str = Field(description="商品标题，符合亚马逊跨境规范，长度控制在60-120字符")
    bullet_points: list[str] = Field(..., min_length=5, max_length=5, description="五点描述，数组，固定5条卖点")
    ad_copy: str = Field(description="简短广告文案，适合商品详情首段介绍")


def parse_listing_output(raw_text: str, trace_id: str) -> Optional[ListingOutputSchema]:
    """
    Listing专用解析器：JSON解析 + Pydantic字段校验 + 业务规则二次校验
    """
    json_data = safe_parse_json(raw_text, trace_id)
    if json_data is None:
        logger.error(f"[parse_listing_output][trace_id={trace_id}] Listing JSON解析失败")
        return None
    try:
        validated = ListingOutputSchema(**json_data)
    except Exception as e:
        logger.error(f"[parse_listing_output][trace_id={trace_id}] Listing字段校验失败: {str(e)}")
        return None

    # ========== 新增业务专项校验（不改动上面原有逻辑） ==========
    error_reasons = []
    # 标题长度 60~120
    title_len = len(validated.title)
    if not (60 <= title_len <= 120):
        error_reasons.append(f"标题长度{title_len}，要求60~120字符")
    # 广告文案最大80字符
    if len(validated.ad_copy) > 80:
        error_reasons.append(f"广告文案超长，最大80字符")
    # 检测中文
    chinese_reg = re.compile(r'[\u4e00-\u9fff]')
    if chinese_reg.search(validated.title):
        error_reasons.append("标题包含中文，必须全英文")
    if chinese_reg.search(validated.ad_copy):
        error_reasons.append("广告文案包含中文，必须全英文")
    for idx, bullet in enumerate(validated.bullet_points):
        if bullet.strip() == "":
            error_reasons.append(f"第{idx+1}条五点描述为空")
        if chinese_reg.search(bullet):
            error_reasons.append(f"第{idx+1}条五点描述包含中文")

    if error_reasons:
        err_msg = "；".join(error_reasons)
        logger.warning(f"[parse_listing_output][trace_id={trace_id}] 业务规则校验失败：{err_msg}")
        return None

    return validated


# ===================== 销售查询参数解析Schema =====================
class SalesQueryParamSchema(BaseModel):
    product_id: Optional[str] = Field(default=None, description="商品ID/SKU")
    month: Optional[str] = Field(default=None, description="月份，严格YYYY-MM格式")


def parse_sales_params(raw_text: str, trace_id: str) -> Optional[SalesQueryParamSchema]:
    """销售查询自然语言参数解析结果校验：JSON解析 + Pydantic + 月份格式校验"""
    json_data = safe_parse_json(raw_text, trace_id)
    if json_data is None:
        logger.error(f"[parse_sales_params][trace_id={trace_id}] sales params json parse failed")
        return None
    try:
        validated = SalesQueryParamSchema(**json_data)
    except Exception as e:
        logger.error(f"[parse_sales_params][trace_id={trace_id}] sales params validate error: {str(e)}")
        return None

    # 新增月份格式校验 YYYY-MM
    month_reg = re.compile(r"^\d{4}-\d{2}$")
    if validated.month is not None and not month_reg.match(validated.month):
        logger.warning(f"[parse_sales_params][trace_id={trace_id}] 月份格式错误，要求YYYY-MM，输入={validated.month}")
        return None

    return validated


# ===================== 模块自测 =====================
if __name__ == "__main__":
    print("==== output_parsers 模块自测 ====")
    from utils.helper import generate_trace_id
    test_trace_id = generate_trace_id()
    # case1 带```json代码块
    case1 = """```json {"task_list":["任务1","任务2"]} ```"""
    res1 = safe_parse_json(case1, test_trace_id)
    print(f"case1解析结果: {res1}")

    # case2 纯json字符串
    case2 = '{"name":"test"}'
    res2 = safe_parse_json(case2, test_trace_id)
    print(f"case2解析结果: {res2}")

    # case3 损坏JSON
    case3 = '{bad json}'
    res3 = safe_parse_json(case3, test_trace_id)
    print(f"case3解析结果: {res3}")

    # case4 Listing校验
    case4 = """```json
    {
        "title": "Wireless Bluetooth Headset Noise Cancelling",
        "bullet_points": ["Noise cancelling", "Long battery", "Stable connection", "Lightweight", "IP54"],
        "ad_copy": "Perfect bluetooth earphone for travel and daily use."
    }
    ```"""
    res4 = parse_listing_output(case4, test_trace_id)
    print(f"case4 Listing校验结果: {res4}")

    # case5 sales params
    case5 = '{"product_id":"SKU001","month":"2026-09"}'
    res5 = parse_sales_params(case5, test_trace_id)
    print(f"case5 sales params: {res5}")

    print("✅ output_parsers自测完成")
