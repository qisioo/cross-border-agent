FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 使用清华PyPI镜像源安装uv，增加超时时间
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=120 uv

COPY pyproject.toml uv.lock ./

# uv也指定清华源做依赖安装
RUN uv sync --frozen --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "main"]
