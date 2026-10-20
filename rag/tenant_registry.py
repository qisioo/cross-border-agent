""" 租户检索注册表 tenant_registry.py
文件作用：按租户(tenant_id)管理各自独立的 HybridRetriever 实例，实现知识库隔离。
每个租户有自己独立的 Milvus collection(agent03_<tenant_id>) 与内存BM25索引，
检索时只取本租户的检索器，保证不同企业知识库互不污染。
线程安全：多租户并发上传/构建/检索时，读写用锁保护。
"""
import threading
from typing import Dict, Optional
from rag.hybrid_retriever import HybridRetriever


class TenantRetrieverRegistry:
    """按租户维护 {tenant_id -> HybridRetriever} 的注册表，隔离各企业知识库"""

    def __init__(self):
        self._map: Dict[str, HybridRetriever] = {}
        self._lock = threading.Lock()

    def register(self, tenant_id: str, retriever: HybridRetriever) -> None:
        """注册/覆盖某租户的检索器（构建完成后调用）"""
        with self._lock:
            self._map[tenant_id] = retriever
        print(f"[TenantRetrieverRegistry] 注册租户检索器: {tenant_id}")

    def get(self, tenant_id: str) -> Optional[HybridRetriever]:
        """取某租户的检索器，未注册返回 None"""
        with self._lock:
            return self._map.get(tenant_id)

    def has(self, tenant_id: str) -> bool:
        """判断某租户是否已构建检索器"""
        with self._lock:
            return tenant_id in self._map

    def remove(self, tenant_id: str) -> None:
        """移除某租户检索器（如重建/删除租户时）"""
        with self._lock:
            self._map.pop(tenant_id, None)
        print(f"[TenantRetrieverRegistry] 移除租户检索器: {tenant_id}")

    def tenant_ids(self) -> list:
        """返回已注册的租户 id 列表"""
        with self._lock:
            return list(self._map.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)


# 全局单例：main.py 启动时创建，agent 图与上传接口共用
_registry_instance: Optional[TenantRetrieverRegistry] = None


def get_tenant_registry() -> TenantRetrieverRegistry:
    """获取全局租户检索注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TenantRetrieverRegistry()
    return _registry_instance
