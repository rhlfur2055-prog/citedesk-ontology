FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY web ./web
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" CITEDESK_USE_FAKE_LLM=true
EXPOSE 8000
CMD ["sh", "-c", "citedesk ingest data/corpus && citedesk serve --host 0.0.0.0 --port 8000"]
