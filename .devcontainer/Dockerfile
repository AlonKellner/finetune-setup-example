FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /home/vscode/.cache && chmod 777 /home/vscode/.cache
