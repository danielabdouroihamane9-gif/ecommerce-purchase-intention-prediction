FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --python 3.12 --no-dev

COPY . .
RUN uv run python train.py

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
