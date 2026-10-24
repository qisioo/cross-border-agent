""" 对话元数据存储 conversation_store.py
文件作用：按企业(tenant_id)管理其下多个对话(conversation)，实现"企业主ID → 多对话"结构。
- 每个对话有唯一 id(创建时自动生成，8位hex) 与可编辑 name(显示名，支持中文)
- 对话之间完全隔离：知识库目录/向量库/检索均按对话独立，绝不跨对话
- 对话元数据持久化到 rag/knowledge/<tenant_id>/conversations.json，线程安全(全局锁)
注意：对话的知识库文件/向量库清理由调用方(main.py)负责，本模块只管理对话元数据。
"""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# 知识库根目录（与 rag/build_vector_db.py 的 KNOWLEDGE_FOLDER 一致）
KNOWLEDGE_ROOT = Path(__file__).parent / "knowledge"

# 全局锁：保护所有企业的对话元数据文件读写（并发创建/改名/删除安全）
_store_lock = threading.Lock()


def _conv_file(tenant_id: str) -> Path:
    """某企业对话元数据文件路径 rag/knowledge/<tenant_id>/conversations.json"""
    return KNOWLEDGE_ROOT / tenant_id / "conversations.json"


def _load(tenant_id: str) -> List[dict]:
    """读取某企业全部对话元数据，文件不存在或损坏返回空列表"""
    f = _conv_file(tenant_id)
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        convs = data.get("conversations", []) if isinstance(data, dict) else data
        return convs if isinstance(convs, list) else []
    except Exception:
        return []


def _save(tenant_id: str, convs: List[dict]) -> None:
    """持久化某企业对话元数据到 conversations.json"""
    f = _conv_file(tenant_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        json.dumps({"conversations": convs}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def _gen_id() -> str:
    """生成对话唯一id（8位hex，用于目录/collection命名，只能字母数字）"""
    return uuid.uuid4().hex[:8]


def list_conversations(tenant_id: str) -> List[dict]:
    """列出某企业全部对话（按创建顺序）"""
    with _store_lock:
        return list(_load(tenant_id))


def get_conversation(tenant_id: str, conversation_id: str) -> Optional[dict]:
    """取某企业下指定对话，不存在返回None"""
    with _store_lock:
        for c in _load(tenant_id):
            if c.get("id") == conversation_id:
                return c
    return None


def create_conversation(tenant_id: str, name: str = "") -> dict:
    """创建对话：自动生成id；name为空用默认名"对话N"；同时创建对话专属知识库目录"""
    with _store_lock:
        convs = _load(tenant_id)
        conv_id = _gen_id()
        ids = {c.get("id") for c in convs}
        while conv_id in ids:  # 保证id唯一
            conv_id = _gen_id()
        default_name = (name or "").strip() or f"对话{len(convs) + 1}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conv = {"id": conv_id, "name": default_name, "created_at": now, "updated_at": now}
        convs.append(conv)
        _save(tenant_id, convs)
        # 创建对话专属知识库目录 rag/knowledge/<tenant_id>/<conversation_id>/（文件层隔离）
        (KNOWLEDGE_ROOT / tenant_id / conv_id).mkdir(parents=True, exist_ok=True)
        return conv


def rename_conversation(tenant_id: str, conversation_id: str, name: str) -> Optional[dict]:
    """改对话显示名，返回更新后对话；对话不存在返回None"""
    with _store_lock:
        convs = _load(tenant_id)
        for c in convs:
            if c.get("id") == conversation_id:
                c["name"] = (name or "").strip() or c["name"]
                c["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save(tenant_id, convs)
                return c
    return None


def delete_conversation(tenant_id: str, conversation_id: str) -> bool:
    """删除对话元数据；返回是否删除成功（知识库文件/向量库由main清理）"""
    with _store_lock:
        convs = _load(tenant_id)
        new_convs = [c for c in convs if c.get("id") != conversation_id]
        if len(new_convs) == len(convs):
            return False
        _save(tenant_id, new_convs)
        return True
