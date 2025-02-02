FROM nvcr.io/nvidia/pytorch:25.01-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile --extra mms_blog_post pyproject.toml -o requirements.txt && \
    uv sync --extra mms_blog_post

ENTRYPOINT [ "/app/.venv/bin/python3" ]
