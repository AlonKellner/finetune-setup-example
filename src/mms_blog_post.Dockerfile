ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-runtime-ubuntu24.04
FROM $BASE_IMAGE AS base
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx
ENV PATH="/root/.local/bin/:$PATH"
RUN uv self update

WORKDIR /app

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:$LD_LIBRARY_PATH

RUN --mount=type=cache,dst=/root/.cache/ \
    echo general-clean && rm -rf /opt/conda && rm -rf /var/lib/apt/lists/* && apt clean && \
    echo conda-check && if [ ! -f /root/.cache/miniconda.sh ]; then \
      echo conda-down && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/.cache/miniconda.sh; \
    fi && \
    echo conda-setup && /bin/bash /root/.cache/miniconda.sh -b -p /opt/conda && \
    echo conda-tools && conda install -y -c conda-forge 'ffmpeg<7,>5' sox libstdcxx-ng libgcc ncurses && ffmpeg -version && sox --version && \
    echo apt-setup && apt update && apt upgrade -y && \
    echo apt-tools && apt install -y --no-install-recommends bash ca-certificates curl file git \
    inotify-tools jq libgl1 lsof vim nano tmux nginx openssh-server procps \
    rsync sudo software-properties-common unzip wget zip

COPY dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

ARG IS_ROCM
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=${IS_ROCM}
ARG EXTRA_GROUP
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv pip compile pyproject.toml --group mms_blog_post --extra ${EXTRA_GROUP} --extra flash-attn -o requirements.txt && \
    uv sync --no-default-groups --group mms_blog_post --extra ${EXTRA_GROUP} && \
    MAX_JOBS=4 uv sync --no-default-groups --group mms_blog_post --extra ${EXTRA_GROUP} --extra flash-attn

ARG WORKDIR=/app
WORKDIR ${WORKDIR}
RUN echo "Some bug makes the last RUN action to be not cached, so this is a workaround"

FROM base AS python
ENTRYPOINT [ "/app/.venv/bin/python" ]

FROM base AS bare
ENTRYPOINT []
