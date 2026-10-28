"""触发知识库重建"""
import requests

url = "http://127.0.0.1:8000/api/tenant/default/conversations/8c6fd7b3/rebuild"
response = requests.post(url)
print(f"状态码：{response.status_code}")
print(f"结果：{response.text}")
