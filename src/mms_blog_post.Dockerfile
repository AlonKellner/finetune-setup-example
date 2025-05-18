FROM nvcr.io/nvidia/cuda-dl-base:25.02-cuda12.8-devel-ubuntu24.04
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/bin/uvx && \
    chmod 777 /root/.local/bin/uv && \
    chmod 777 /root/.local/bin/uvx && \
    uv self update

WORKDIR /app

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:$LD_LIBRARY_PATH

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
    uv sync --no-default-groups --group mms_blog_post --upgrade && \
    uv sync --no-default-groups --group mms_blog_post_gpu --upgrade --no-build-isolation-package flash-attn

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python" ]
