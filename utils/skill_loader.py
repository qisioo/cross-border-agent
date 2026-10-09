"""
utils/skill_loader.py
文件作用：Skill 规则库渐进式加载器。按分支（task_type）只加载 rules/ 下对应分片，避免读取整个 SKILL.md 浪费 token；
启动时预加载全部分片进内存，运行时零 IO。使用模块级单例，供 tools.py / nodes.py 直接调用，不改变函数签名与返回结构。
"""
from pathlib import Path
from typing import Optional, Dict

# skill 根目录（相对本项目根目录）
SKILL_DIR = Path(__file__).resolve().parent.parent / "skill"
RULES_DIR = SKILL_DIR / "rules"


class SkillLoader:
    """Skill 规则加载器：预加载全部规则分片，按分支名取用"""

    def __init__(self, rules_dir: Path = RULES_DIR):
        self.rules_dir = rules_dir
        self._rules: Dict[str, str] = {}
        self.preload_rules()

    def preload_rules(self) -> None:
        """启动阶段预加载 rules/ 下所有 .md 分片进内存，运行时零 IO"""
        self._rules = {}
        if not self.rules_dir.exists():
            print(f"[SkillLoader][警告] 规则目录不存在：{self.rules_dir}")
            return
        for md_file in sorted(self.rules_dir.glob("*.md")):
            branch = md_file.stem  # 文件名即分支名，如 listing、sales
            try:
                self._rules[branch] = md_file.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[SkillLoader][警告] 读取规则失败 {md_file.name}: {str(e)}")
        print(f"[SkillLoader] 规则预加载完成，共加载 {len(self._rules)} 个分支规则：{sorted(self._rules.keys())}")

    def load_rule(self, branch: str) -> Optional[str]:
        """按分支名取规则文本；未预加载或不存在返回 None"""
        return self._rules.get(branch)


# ===================== 模块级单例 =====================
_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """
    获取 SkillLoader 单例。首次调用自动初始化（懒加载），
    main.py lifespan 中也可显式初始化以提前预加载。
    """
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader


def reset_skill_loader() -> None:
    """重置单例（测试用），清空已缓存规则"""
    global _loader
    _loader = None


# ===================== 自测代码 =====================
if __name__ == "__main__":
    print("==== utils.skill_loader 自测 ====")
    loader = get_skill_loader()
    print(f"已加载分支: {sorted(loader._rules.keys())}")
    for branch in ["listing", "sales", "customer_service", "result_collect"]:
        rule = loader.load_rule(branch)
        print(f"[{branch}] 规则存在={rule is not None}, 长度={len(rule) if rule else 0}")
    print("✅ skill_loader 自测完成")
