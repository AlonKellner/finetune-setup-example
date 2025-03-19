FROM nvcr.io/nvidia/cuda-dl-base:25.02-cuda12.8-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN --mount=type=cache,dst=/root/.cache/ \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && \
    conda install -c conda-forge 'ffmpeg<7,>5' sox && ffmpeg -version && sox --version

COPY uv.toml ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

COPY pyproject.toml uv.lock dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml --group mms_blog_post_gpu -o requirements.txt && \
    uv sync --group mms_blog_post --upgrade --link-mode=copy && \
    uv sync --group mms_blog_post_gpu --upgrade --link-mode=copy --no-build-isolation-package flash-attn

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python" ]
