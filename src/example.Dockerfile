FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml README.md ./
COPY src/_version.py src/
RUN --mount=type=cache,target=/root/.cache/ \
    uv sync --all-extras
