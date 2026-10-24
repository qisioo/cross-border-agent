""" Agent03 对话级隔离工作台 —— Streamlit 前端
结构：企业(tenant_id) → 多个对话(conversation) → 每对话独立知识库
功能：
- 企业选择：输入 tenant_id，企业下可建多个对话
- 对话管理：新建/列表/改名/删除（企业主ID下的多对话，相互完全隔离）
- 对话知识库管理：上传/列表/删除/重建（每对话独立向量库 agent03_<tenant>_<conv>）
- 智能问答：多轮对话，仅基于当前对话绑定的知识库作答，绝不跨对话
后端接口：http://127.0.0.1:8000（FastAPI main.py）
"""
import time
import urllib.parse
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="跨境电商智能运营 Agent 工作台", page_icon="🤖", layout="wide")
st.title("🤖 跨境电商智能运营 Agent 工作台")

# ---------- 初始化会话状态 ----------
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "default"
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 侧边栏：企业管理 + 对话管理 + 当前对话知识库 ==========
with st.sidebar:
    st.header("🏢 企业管理")
    tenant_id = st.text_input("企业ID (tenant_id)", value=st.session_state.tenant_id,
                              help="每个企业独立，企业下可建多个对话，互不污染")
    tenant_id = (tenant_id or "default").strip() or "default"
    # 企业ID变化时重置当前对话与历史
    if tenant_id != st.session_state.tenant_id:
        st.session_state.tenant_id = tenant_id
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
    st.session_state.tenant_id = tenant_id

    st.divider()

    # ===== 对话管理（企业主ID下多对话） =====
    st.subheader("💬 对话管理")
    new_conv_name = st.text_input("新对话名称（可选）", placeholder="如：国内商品 / 国外商品")
    if st.button("➕ 新建对话", use_container_width=True):
        try:
            body = {"name": new_conv_name} if (new_conv_name or "").strip() else {}
            r = requests.post(f"{API_BASE}/api/tenant/{tenant_id}/conversations", json=body, timeout=15)
            if r.status_code == 200:
                conv = r.json()["data"]
                st.session_state.current_conversation_id = conv["id"]
                st.session_state.messages = []
                st.success(f"已创建对话：{conv['name']}（{conv['id']}）")
                st.rerun()
            else:
                st.error(f"创建失败: {r.text}")
        except Exception as e:
            st.error(f"请求失败: {e}")

    # 列出对话
    try:
        r = requests.get(f"{API_BASE}/api/tenant/{tenant_id}/conversations", timeout=15)
        convs = r.json()["data"] if r.status_code == 200 else []
    except Exception:
        convs = []
    if convs:
        options = [f"{c['name']}（{c['id']}）" for c in convs]
        cur_idx = 0
        if st.session_state.current_conversation_id:
            for i, c in enumerate(convs):
                if c["id"] == st.session_state.current_conversation_id:
                    cur_idx = i
                    break
        sel = st.selectbox("选择对话", options, index=cur_idx, key="conv_select")
        sel_id = convs[options.index(sel)]["id"]
        if sel_id != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = sel_id
            st.session_state.messages = []

        # 当前对话操作：改名 / 删除
        ctrl = st.radio("对话操作", ["无", "改名", "删除"], horizontal=True, key="conv_ctrl")
        if ctrl == "改名":
            new_name = st.text_input("新名称", key="rename_input")
            if st.button("✅ 保存改名"):
                try:
                    r = requests.put(
                        f"{API_BASE}/api/tenant/{tenant_id}/conversations/{sel_id}",
                        json={"name": new_name}, timeout=15)
                    if r.status_code == 200:
                        st.success("改名成功")
                        st.rerun()
                    else:
                        st.error(f"改名失败: {r.text}")
                except Exception as e:
                    st.error(f"请求失败: {e}")
        elif ctrl == "删除":
            st.warning("删除对话将同时清除其知识库文件与向量库，不可恢复！")
            if st.button("🗑️ 确认删除对话"):
                try:
                    r = requests.delete(
                        f"{API_BASE}/api/tenant/{tenant_id}/conversations/{sel_id}", timeout=15)
                    if r.status_code == 200:
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                        st.success("对话已删除")
                        st.rerun()
                    else:
                        st.error(f"删除失败: {r.text}")
                except Exception as e:
                    st.error(f"请求失败: {e}")
    else:
        st.info("该企业暂无对话，请先新建。")

    st.divider()

    # ===== 当前对话知识库管理 =====
    conv_id = st.session_state.current_conversation_id
    if conv_id:
        st.subheader(f"📚 对话知识库（{conv_id[:8]}）")
        # 上传到当前对话
        uploaded_file = st.file_uploader("上传 txt/md/csv", type=["txt", "md", "csv"], key="kb_upload")
        if uploaded_file is not None:
            if st.button("开始上传并构建", type="primary", use_container_width=True):
                try:
                    with st.spinner("上传文档中..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        resp = requests.post(
                            f"{API_BASE}/api/tenant/{tenant_id}/conversations/{conv_id}/upload",
                            files=files, timeout=30)
                    if resp.status_code != 200:
                        st.error(f"上传失败: {resp.text}")
                    else:
                        task_id = resp.json()["task_id"]
                        bar = st.progress(0, text="正在构建向量数据库...")
                        while True:
                            p = requests.get(f"{API_BASE}/api/tenant/build/{task_id}", timeout=10).json()
                            progress = int(p.get("progress", 0))
                            message = p.get("message", "")
                            status = p.get("status", "running")
                            bar.progress(min(max(progress, 0), 100) / 100, text=message)
                            if status in ("done", "failed"):
                                if status == "done":
                                    st.success("✅ 知识库构建完成！")
                                else:
                                    st.error(f"❌ 构建失败: {message}")
                                break
                            time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"请求失败: {e}")

        # 当前对话知识库文件列表 + 删除
        try:
            r = requests.get(
                f"{API_BASE}/api/tenant/{tenant_id}/conversations/{conv_id}/knowledge", timeout=15)
            kb = r.json()["data"] if r.status_code == 200 else {"files": [], "built": False, "chunk_count": 0}
        except Exception:
            kb = {"files": [], "built": False, "chunk_count": 0}
        files = kb.get("files", [])
        built = kb.get("built", False)
        st.caption(f"构建状态：{'✅ 已构建' if built else '❌ 未构建'} | 向量块：{kb.get('chunk_count', 0)}")
        if files:
            st.markdown("**知识库文件：**")
            for f in files:
                col1, col2 = st.columns([4, 1])
                col1.caption(f"📄 {f['name']}（{f['size']}B）")
                if col2.button("删", key=f"del_{f['name']}"):
                    try:
                        r = requests.delete(
                            f"{API_BASE}/api/tenant/{tenant_id}/conversations/{conv_id}"
                            f"/knowledge/{urllib.parse.quote(f['name'])}",
                            timeout=15)
                        if r.status_code == 200:
                            st.success("文件已删除，正在重建...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"删除失败: {r.text}")
                    except Exception as e:
                        st.error(f"请求失败: {e}")
        else:
            st.caption("暂无知识库文件。")

        if st.button("🔄 重建知识库", use_container_width=True):
            try:
                r = requests.post(
                    f"{API_BASE}/api/tenant/{tenant_id}/conversations/{conv_id}/rebuild", timeout=15)
                if r.status_code == 200:
                    task_id = r.json()["task_id"]
                    bar = st.progress(0, text="正在重建...")
                    while True:
                        p = requests.get(f"{API_BASE}/api/tenant/build/{task_id}", timeout=10).json()
                        bar.progress(min(max(int(p.get("progress", 0)), 0), 100) / 100, text=p.get("message", ""))
                        if p.get("status") in ("done", "failed"):
                            if p.get("status") == "done":
                                st.success("重建完成")
                            else:
                                st.error(f"重建失败: {p.get('message')}")
                            break
                        time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"请求失败: {e}")

    st.divider()
    st.caption(f"后端接口：{API_BASE}")

# ========== 主区：智能问答（多轮，基于当前对话） ==========
conv_id = st.session_state.current_conversation_id
conv_name = "未选择对话"
if conv_id:
    try:
        r = requests.get(f"{API_BASE}/api/tenant/{st.session_state.tenant_id}/conversations", timeout=15)
        for c in r.json().get("data", []):
            if c["id"] == conv_id:
                conv_name = c["name"]
                break
    except Exception:
        pass

st.header("💬 智能问答")
st.caption(f"当前企业：`{st.session_state.tenant_id}`　当前对话：`{conv_name}`（{conv_id or '未选择'}）")
if not conv_id:
    st.info("请先在左侧新建或选择一个对话，再开始提问。对话之间知识库完全隔离，绝不跨对话。")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("请输入您的问题", disabled=(not conv_id))
if prompt and conv_id:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    payload = {
        "user_query": prompt,
        "tenant_id": st.session_state.tenant_id,
        "conversation_id": conv_id,
    }
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
