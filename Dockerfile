
# -------- Builder --------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

COPY pyproject.toml uv.lock .python-version ./

#force uv to run python 3.11 and not install any other most recent version
RUN uv sync --frozen --no-dev --no-cache --python /usr/local/bin/python

# Remove caches
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} +


# -------- Runtime --------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY main.py helpers.py ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 9596

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9596"]
