FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv sync

ENTRYPOINT [ "/app/.venv/bin/python3" ]
