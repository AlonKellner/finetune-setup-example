FROM mcr.microsoft.com/devcontainers/base:debian
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY uv.toml  ./
RUN uv python install --preview --default && \
    mkdir /home/vscode/.cache && \
    chmod 777 /home/vscode/.cache
