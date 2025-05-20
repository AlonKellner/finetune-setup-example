FROM nvcr.io/nvidia/cuda-dl-base:25.04-cuda12.9-devel-ubuntu24.04
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx
ENV PATH="/root/.local/bin/:$PATH"
RUN uv self update

WORKDIR /app

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:$LD_LIBRARY_PATH

RUN --mount=type=cache,dst=/root/.cache/ \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && \
    conda install -c conda-forge 'ffmpeg<7,>5' sox && ffmpeg -version && sox --version

COPY dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml --group mms_blog_post --extra gpu -o requirements.txt && \
    MAX_JOBS=2 uv sync --no-default-groups --group mms_blog_post --extra gpu --upgrade

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
ENTRYPOINT [ "/app/.venv/bin/python" ]
