FROM python:3.12.3-slim

#install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

#CREATE env for uv dependencies
ENV PATH="/app/.venv/bin:$PATH"

#copy uv dependency files
COPY pyproject.toml uv.lock .python-version ./


#install uv dependencies
RUN uv sync --locked --no-cache

COPY main.py helpers.py ./


EXPOSE 9596


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9596"]





