""" Agent03 多租户工作台 —— Streamlit 前端
功能：
- 企业(租户)选择：输入 tenant_id，隔离各企业知识库
- 企业知识库上传：上传 txt/md/csv，后端异步构建向量库(agent03_<tenant_id>)，前端实时进度条
- 智能问答：多轮对话，基于所选企业知识库作答
后端接口：http://127.0.0.1:8000（FastAPI main.py）
"""
import time
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="跨境电商智能运营 Agent 工作台", page_icon="🤖", layout="wide")
st.title("🤖 跨境电商智能运营 Agent 工作台")

# ---------- 初始化会话状态（企业ID + 对话历史） ----------
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "default"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 侧边栏：企业管理 + 知识库上传 ==========
with st.sidebar:
    st.header("🏢 企业管理")
    tenant_id = st.text_input("企业ID (tenant_id)", value=st.session_state.tenant_id,
                              help="每个企业独立知识库，互不污染")
    st.session_state.tenant_id = (tenant_id or "default").strip() or "default"

    st.divider()
    st.subheader("📤 上传企业知识库")
    uploaded_file = st.file_uploader("选择 txt/md/csv 文档", type=["txt", "md", "csv"])
    if uploaded_file is not None:
        if st.button("开始上传并构建", type="primary"):
            try:
                with st.spinner("上传文档中..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"tenant_id": st.session_state.tenant_id}
                    resp = requests.post(f"{API_BASE}/api/tenant/upload", data=data, files=files, timeout=30)
                if resp.status_code != 200:
                    st.error(f"上传失败: {resp.text}")
                else:
                    task_id = resp.json()["task_id"]
                    # 实时进度条：轮询后端构建进度
                    bar = st.progress(0, text="正在构建向量数据库...")
                    while True:
                        p = requests.get(f"{API_BASE}/api/tenant/build/{task_id}", timeout=10).json()
                        progress = int(p.get("progress", 0))
                        message = p.get("message", "")
                        status = p.get("status", "running")
                        bar.progress(min(max(progress, 0), 100) / 100, text=message)
                        if status in ("done", "failed"):
                            if status == "done":
                                st.success("✅ 企业知识库构建完成，可开始提问！")
                            else:
                                st.error(f"❌ 构建失败: {message}")
                            break
                        time.sleep(1)
            except Exception as e:
                st.error(f"请求失败: {e}")

    st.divider()
    st.caption(f"后端接口：{API_BASE}")

# ========== 主区：智能问答（多轮） ==========
st.header("💬 智能问答")
st.caption(f"当前企业：`{st.session_state.tenant_id}`")

# 展示历史对话
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("请输入您的问题，例如：企业的大客户返点政策是什么？")
if prompt:
    # 追加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 请求后端 Agent
    payload = {"user_query": prompt, "tenant_id": st.session_state.tenant_id}
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.post(f"{API_BASE}/api/agent/run", json=payload, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("final_answer") or data.get("msg") or "（无回答）"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"请求失败: {resp.text}")
            except Exception as e:
                st.error(f"请求异常: {e}")
