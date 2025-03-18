FROM mcr.microsoft.com/devcontainers/base:debian
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install && \
    mkdir /home/vscode/.cache && \
    chmod 777 /home/vscode/.cache
