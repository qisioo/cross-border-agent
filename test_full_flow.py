"""全流程测试：知识库检索 + 多意图 + 闲聊 + 上下文"""
import requests
import time

BASE_URL = "http://127.0.0.1:8000"
TENANT_ID = "default"
CONV_ID = "8c6fd7b3"

def ask(question, conv_id=CONV_ID):
    """发问题，返回回答"""
    url = f"{BASE_URL}/api/agent/run"
    data = {
        "tenant_id": TENANT_ID,
        "conversation_id": conv_id,
        "user_query": question
    }
    response = requests.post(url, json=data)
    result = response.json()
    return result.get("final_answer", "无回答")

print("=" * 60)
print("测试1：知识库检索（业务问题）")
print("=" * 60)
ans = ask("退换货政策是什么？")
print(f"问：退换货政策是什么？")
print(f"答：{ans[:150]}...")
print()

print("=" * 60)
print("测试2：知识库检索（物流问题）")
print("=" * 60)
ans = ask("全国物流几天能到？")
print(f"问：全国物流几天能到？")
print(f"答：{ans[:150]}...")
print()

print("=" * 60)
print("测试3：闲聊（天气）")
print("=" * 60)
ans = ask("武汉今天天气怎么样？")
print(f"问：武汉今天天气怎么样？")
print(f"答：{ans[:150]}...")
print()

print("=" * 60)
print("测试4：多意图复合问题")
print("=" * 60)
ans = ask("退换货政策是什么？还有现在有什么优惠活动吗？")
print(f"问：退换货政策是什么？还有现在有什么优惠活动吗？")
print(f"答：{ans[:200]}...")
print()

print("=" * 60)
print("测试5：上下文记忆（多轮对话）")
print("=" * 60)
ans1 = ask("我叫张三，住在武汉")
print(f"第一轮：我叫张三，住在武汉")
print(f"答：{ans1[:100]}...")
time.sleep(1)
ans2 = ask("我叫什么名字？住在哪里？")
print(f"第二轮：我叫什么名字？住在哪里？")
print(f"答：{ans2[:150]}...")
print()

print("=" * 60)
print("测试完成")
print("=" * 60)
