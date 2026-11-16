FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --python 3.12 --no-dev

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync uvicorn predict:app --host 0.0.0.0 --port ${PORT:-8000}"]
