<!-- 文件作用：销售数据查询的 B 类字段规范规则分片，由 agent/tools.py 的 sales_data_query 通过 skill_loader 注入给 LLM。仅含参数提取字段规范（模型需遵守的固定规则），不包含 CSV 读取、编码回退、精确匹配等代码内部逻辑（那些属代码内部执行，不注入）。 -->

# 销售查询参数提取规则（注入给 LLM）

从用户提问提取商品 ID（product_id）与月份（month，格式 YYYY-MM），严格遵守以下规则：
1. 没有识别到对应字段时，该字段返回 null。
2. 仅输出 JSON，禁止多余文字、markdown 标记。
3. 输出示例：{"product_id":"SKU001","month":"2026-09"}
