# syntax=docker/dockerfile:1
# 多阶段构建：builder 装依赖，runtime 只带运行所需文件，并以非 root 运行
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.12.10

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SMA_DB_PATH=/data/sma.db \
    SMA_CHROMA_DIR=/data/chroma \
    SMA_KNOWLEDGE_DIR=/app/data/knowledge

RUN useradd --create-home --uid 10001 sma && mkdir -p /data && chown sma:sma /data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=sma:sma src ./src
# 示例知识库随镜像分发，容器启动即可用
COPY --chown=sma:sma data/knowledge ./data/knowledge

USER sma
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "sma.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/dev/null"]
