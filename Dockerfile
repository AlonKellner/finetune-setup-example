FROM mcr.microsoft.com/devcontainers/base:debian
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/bin/uvx && \
    chmod 777 /root/.local/bin/uv && \
    chmod 777 /root/.local/bin/uvx && \
    uv self update

WORKDIR /app

COPY dev-pyproject/ ./

RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml --group dev -o requirements.txt && \
    uv sync --upgrade

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python" ]
