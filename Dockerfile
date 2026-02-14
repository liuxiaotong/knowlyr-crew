# ── Stage 1: Build ──
FROM python:3.13-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/dist

# ── Stage 2: Runtime ──
FROM python:3.13-slim

LABEL maintainer="Liu Kai <mrliukai@gmail.com>"
LABEL description="Crew — AI Skill Loader"

WORKDIR /app

# 安装 wheel + 可选依赖（webhook 模式所需）
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[webhook,execute,openai,id] \
    && rm -f /tmp/*.whl

# 默认项目目录（挂载用户项目到此处）
RUN mkdir -p /app/project

ENV PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

ENTRYPOINT ["knowlyr-crew"]
CMD ["serve", "--port", "8765", "--project-dir", "/app/project"]
