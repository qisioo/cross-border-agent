""" agent/mcp_tools.py —— MCP 工具封装（接入独立 MCP Server）

作用：
    通过 langchain-mcp-adapters（LangChain 官方的 MCP 适配器）连接
    mcp_demo/server.py 这个独立 MCP 服务器，把它的 get_weather / get_time
    工具转成这里可直接调用的函数，供日常对话分支使用。

体现 MCP 的价值：
    工具（查天气/查时间）是独立于 Agent 之外存在的（mcp_demo 里），
    本模块只是扮演"客户端"，按标准协议去调用它。
    将来换模型、换 Agent，这套 MCP Server 照常可复用。

注意：
    本模块只是"调用方"，不承载工具实现。工具实现都在 mcp_demo/server.py。
"""
import asyncio
import os
import re

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# ------------------------------------------------------------
# MCP Server 的位置配置
# server.py 放在 mcp_demo 目录，并且是用 mcp_demo 自己的环境装的 mcp 2.x，
# 所以要用 mcp_demo 的 Python 解释器来启动它，否则会找不到 mcp 库。
# ------------------------------------------------------------
_MCP_DEMO_DIR = r"D:\qi_agent\mcp_demo"
_MCP_SERVER_PY = os.path.join(_MCP_DEMO_DIR, "server.py")
_MCP_PYTHON = os.path.join(_MCP_DEMO_DIR, ".venv", "Scripts", "python.exe")


def _client_config() -> dict:
    """返回 MultiServerMCPClient 需要的服务器配置。"""
    return {
        # 服务器名字（随便起，用于标识）
        "agent03-mcp-tools": {
            "command": _MCP_PYTHON,       # 用哪个解释器启动 server
            "args": [_MCP_SERVER_PY],     # 启动 server.py
            "transport": "stdio",         # 通过标准输入输出通信
        },
    }


async def _call_tool(name: str, args: dict) -> str:
    """异步执行一次 MCP 工具调用（连接 server → 拿工具 → 调用 → 返回文本）。

    注意：langchain-mcp-adapters 0.1.0+ 不能用 `async with MultiServerMCPClient`，
    正确姿势是用 client.session(服务器名) 拿到已初始化的会话，
    再用 load_mcp_tools(session) 把 MCP 工具转成 LangChain 工具列表。
    """
    client = MultiServerMCPClient(_client_config())
    async with client.session("agent03-mcp-tools") as session:
        tools = await load_mcp_tools(session)
        tool = next((t for t in tools if t.name == name), None)
        if tool is None:
            return f"MCP 工具 {name} 不可用（服务器上没这个工具）"
        try:
            result = await tool.ainvoke(args)
            # 工具返回的是 MCP 的 content 列表（[{type,text,id}...]），
            # 只把其中的文本抽出来拼成干净字符串
            if isinstance(result, list):
                parts = [
                    item.get("text", "")
                    for item in result
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"MCP 工具调用失败：{e}"


def get_weather(city: str) -> str:
    """调用 MCP 天气工具，返回指定城市的天气文本。"""
    return asyncio.run(_call_tool("get_weather", {"city": city}))


def get_time(timezone: str = "Asia/Shanghai") -> str:
    """调用 MCP 时间工具，返回指定时区的当前时间文本。"""
    return asyncio.run(_call_tool("get_time", {"timezone": timezone}))


# ------------------------------------------------------------
# 日常对话识别：判断用户这句话需不需要调用天气 / 时间工具
# 用简单关键词 + 城市提取，够用即可，不用上复杂 NLU。
# ------------------------------------------------------------
_WEATHER_PAT = re.compile(r"天气|气温|温度|下雨|下雪|天冷|天热|多少度|会不会[雨雪]|风力|湿度")
_TIME_PAT = re.compile(r"几点|什么时间|现在时间|当前时间|几点了|时区|几月几号")

# 常见城市表：识别时优先精确匹配，命中率最高、最稳
_COMMON_CITIES = [
    "北京", "上海", "广州", "深圳", "武汉", "杭州", "南京", "成都", "重庆",
    "西安", "苏州", "天津", "长沙", "郑州", "青岛", "厦门", "合肥", "沈阳",
    "福州", "昆明", "哈尔滨", "长春", "大连", "石家庄", "太原", "南昌",
    "济南", "南宁", "贵阳", "兰州", "西宁", "银川", "乌鲁木齐", "呼和浩特", "拉萨",
]


def _extract_city(user_query: str) -> str:
    """从用户话里尽量抠出城市名，抠不到就默认武汉。"""
    # 方式一：常见城市名直接命中（最可靠）
    for city in _COMMON_CITIES:
        if city in user_query:
            return city
    # 方式二：兜底识别"XX市天气"这种写法
    m = re.search(r"([\u4e00-\u9fa5]{1,3}市)(?:的)?(?:天气|气温|温度)", user_query)
    if m:
        city = m.group(1).replace("市", "")
        if 1 < len(city) <= 3:
            return city
    return "武汉"


def detect_tool_need(user_query: str):
    """检测用户这句话是否需要天气/时间工具。

    返回 (工具类型, 参数字典) 或 None。
    工具类型: "weather" / "time"。
    """
    if not user_query:
        return None
    if _WEATHER_PAT.search(user_query):
        city = _extract_city(user_query)
        return ("weather", {"city": city})
    if _TIME_PAT.search(user_query):
        return ("time", {})
    return None


# ===================== 模块自测 =====================
if __name__ == "__main__":
    print("==== agent.mcp_tools 模块自测 ====")
    # 1) 识别逻辑
    for q in ["武汉今天天气怎么样", "现在几点了", "上海的气温", "帮我查下北京的天气"]:
        need = detect_tool_need(q)
        print(f"  query=「{q}」 -> 识别结果: {need}")
    # 2) 真实调用 MCP（会启动 mcp_demo 的 server）
    print("\n==== 调用 MCP get_weather（武汉） ====")
    print(get_weather("武汉"))
    print("\n==== 调用 MCP get_time ====")
    print(get_time())
