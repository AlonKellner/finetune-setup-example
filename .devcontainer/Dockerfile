FROM mcr.microsoft.com/devcontainers/base:debian
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx
ENV PATH="/home/vscode/.local/bin/:$PATH"
RUN uv self update

USER root

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:${LD_LIBRARY_PATH:-/opt/conda/lib}

COPY dev-pyproject/ ./
RUN echo uv-python && uv python install --preview --default && \
    mkdir /home/vscode/.cache && \
    chmod 777 /home/vscode/.cache && \
    echo general-clean && rm -rf /opt/conda && rm -rf /var/lib/apt/lists/* && apt clean && \
    echo conda-down && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ~/miniconda.sh && \
    echo conda-setup && /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && \
    echo conda-tools && conda install -y -c conda-forge 'ffmpeg<7,>5' libstdcxx-ng libgcc ncurses && ffmpeg -version && \
    echo apt-tools && apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl pkg-config cmake && \
    echo gcloud && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin && \
    echo taplo && curl -fsSL https://github.com/tamasfe/taplo/releases/latest/download/taplo-linux-$(uname -m).gz \
    | gzip -d - | install -m 755 /dev/stdin /usr/local/bin/taplo && taplo --version
