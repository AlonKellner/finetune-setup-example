FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile --all-extras pyproject.toml -o requirements.txt && \
    uv sync --all-extras

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python3" ]
