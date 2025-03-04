FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv sync --upgrade --link-mode=copy

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python3" ]
