FROM mcr.microsoft.com/devcontainers/base:debian
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.toml uv.lock dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default && \
    uv pip compile pyproject.toml --group dev -o requirements.txt && \
    uv sync --upgrade --link-mode=copy

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "python" ]
