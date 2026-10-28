from pymilvus import MilvusClient

client = MilvusClient(uri='http://127.0.0.1:19530')
collection_name = 'agent03_default_8c6fd7b3'

# 查看 collection 信息
desc = client.describe_collection(collection_name)
print('Collection schema:')
for field in desc['fields']:
    print(f'  - {field["name"]}: {field["type"]}')

# 查询前几条数据
res = client.query(collection_name, filter='', output_fields=['text', 'source'], limit=10)
print(f'\n前 {len(res)} 条数据:')
for i, doc in enumerate(res):
    print(f'\n[{i+1}] source: {doc.get("source", "unknown")}')
    print(f'    text: {doc.get("text", "")[:150]}...')
