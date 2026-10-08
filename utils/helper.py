"""
通用工具函数
1.trace_id生成，支持配置前缀
2.封装agent_state更新工具：增加字段合法性校验，过滤非法字段，debug日志输出，规避LangGraph嵌套Pydantic原地修改bug
3.文本工具：支持自定义截断后缀；字符截断（中文按字符计数）
4.会话摘要函数：增加开关，占位模式可告警提示
5.字典工具：支持嵌套字典取值；提供简短别名
6.通用异常包装
"""
import os
import uuid
from typing import Dict, Any, Optional, List
from agent.state import AgentState, GraphState

# 全局配置
TRACE_ID_PREFIX = "trace-"
DUMMY_SUMMARY_RAISE_WARNING = os.getenv("AGENT_DUMMY_SUMMARY_WARN", "1") == "1"


def generate_trace_id() -> str:
    """生成会话追踪trace_id，使用可配置前缀"""
    return f"{TRACE_ID_PREFIX}{uuid.uuid4()}"


def update_agent_state(
    graph_state: GraphState,
    **kwargs
) -> Dict[str, AgentState]:
    """
    封装嵌套AgentState更新逻辑
    - 校验传入字段是否属于AgentState，过滤非法字段并打印警告
    - 输出debug日志打印变更字段
    :param graph_state: LangGraph顶层GraphState
    :param kwargs: 需要更新的AgentState字段键值对
    :return: {"agent_state": new_agent_state}，直接作为节点返回值
    """
    old_agent: AgentState = graph_state["agent_state"]
    valid_fields = set(AgentState.model_fields.keys())

    update_dict: Dict[str, Any] = {}
    invalid_keys: List[str] = []

    for k, v in kwargs.items():
        if k in valid_fields:
            update_dict[k] = v
        else:
            invalid_keys.append(k)

    if invalid_keys:
        print(f"[WARN update_agent_state] 过滤非法字段：{invalid_keys}，AgentState有效字段：{sorted(valid_fields)}")

    print(f"[DEBUG update_agent_state] 变更字段: {list(update_dict.keys())}")
    new_agent = old_agent.model_copy(update=update_dict)
    return {"agent_state": new_agent}


def safe_truncate_text(text: str, max_len: int = 1200, suffix: str = "……[内容截断]") -> str:
    """
    安全文本截断，按字符计数（中文、英文统一字符），支持自定义截断后缀
    :param text:原始文本
    :param max_len:最大保留字符数
    :param suffix:截断追加后缀
    :return:处理后文本
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def dummy_conversation_summary(messages_text: str) -> str:
    """
    会话摘要占位函数
    生产环境必须替换为LLM真实摘要逻辑；环境变量 AGENT_DUMMY_SUMMARY_WARN=1 时输出警告提示
    """
    if not messages_text.strip():
        return ""
    if DUMMY_SUMMARY_RAISE_WARNING:
        print("[WARN dummy_conversation_summary] 当前使用占位摘要逻辑，请替换为真实LLM摘要实现")
    return f"[会话摘要占位] {safe_truncate_text(messages_text, max_len=300)}"


def safe_get(data: Optional[Any], keys: List[str], default: Any = None):
    """
    嵌套字典安全取值，支持多级路径
    example: safe_get(data, ["a","b","c"], default=None) 获取 data["a"]["b"]["c"]
    :param data: 原始字典/对象
    :param keys: 层级key列表
    :param default: 获取失败返回默认值
    :return:目标值或default
    """
    current = data
    for k in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


# 别名，兼容单层快速调用
safe_get_dict_field = lambda d, key, default=None: safe_get(d, [key], default)


# ===================== 内嵌自测代码 =====================
if __name__ == "__main__":
    print("==== utils.helper 工具模块自测 ====")
    from langchain_core.messages import HumanMessage

    # 1.测试trace_id生成
    trace_id = generate_trace_id()
    print(f"生成trace_id: {trace_id}")

    # 2.测试update_agent_state：包含合法字段+非法字段
    init_agent = AgentState(
        user_query="查询蓝牙耳机运营方案",
        trace_id=trace_id
    )
    graph_state = GraphState(
        messages=[HumanMessage(content="查询蓝牙耳机运营方案")],
        agent_state=init_agent
    )

    ret = update_agent_state(
        graph_state,
        product_raw_info={"title":"Buds Pro","price":39.99},
        error_msg="",
        conversation_summary="用户询问蓝牙耳机相关业务",
        fake_field_xxx=12345  # 非法字段，应当被过滤并告警
    )
    new_agent = ret["agent_state"]
    print(f"更新后 product_raw_info: {new_agent.product_raw_info}")
    print(f"更新后 conversation_summary: {new_agent.conversation_summary}")

    # 3.文本截断测试：自定义后缀
    long_text = "测试文本" * 500
    truncated1 = safe_truncate_text(long_text, max_len=200)
    truncated2 = safe_truncate_text(long_text, max_len=200, suffix="【已截断】")
    print(f"\n原始长度:{len(long_text)}, 默认后缀截断长度:{len(truncated1)}")
    print(f"自定义后缀截断结果示例：{truncated2[-30:]}")

    # 4.嵌套字典取值测试
    mock_nested = {"a":{"b":{"c":999}}}
    val_nested = safe_get(mock_nested, ["a","b","c"])
    val_not_exist = safe_get(mock_nested, ["a","x","c"], default=-1)
    val_single = safe_get_dict_field(mock_nested, "a")
    print(f"\n嵌套取值 a.b.c = {val_nested}")
    print(f"不存在路径取值 = {val_not_exist}")
    print(f"单层别名取值 a = {val_single}")

    # 5.占位摘要告警测试
    summary_out = dummy_conversation_summary("用户历史对话内容"*20)
    print(f"\ndummy摘要输出：{summary_out[:100]}")

    print("\n✅ utils.helper自测完成")
    print("提示：节点中直接 return update_agent_state(state, xxx=yyy)，规避嵌套对象原地修改坑")
