FROM nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-runtime-ubuntu24.04 AS base
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
    conda install -c conda-forge 'ffmpeg<7,>5' sox libstdcxx-ng libgcc ncurses && ffmpeg -version && sox --version && \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends bash ca-certificates curl file git \
    inotify-tools jq libgl1 lsof vim nano tmux nginx openssh-server procps \
    rsync sudo software-properties-common unzip wget zip

COPY dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml --group mms_blog_post --extra gpu -o requirements.txt && \
    MAX_JOBS=4 uv sync --no-default-groups --group mms_blog_post && \
    MAX_JOBS=4 uv sync --no-default-groups --group mms_blog_post --extra gpu

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
RUN echo "Some bug makes the last RUN action to be not cached, so this is a workaround"

FROM base AS python
ENTRYPOINT [ "/app/.venv/bin/python" ]

FROM base AS bare
ENTRYPOINT []
