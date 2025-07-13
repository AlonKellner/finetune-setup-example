FROM mcr.microsoft.com/devcontainers/base:debian
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx
ENV PATH="/home/vscode/.local/bin/:$PATH"
RUN uv self update

USER root

COPY dev-pyproject/ ./
RUN uv python install --preview --default && \
    mkdir /home/vscode/.cache && \
    chmod 777 /home/vscode/.cache && \
    apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl pkg-config cmake && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin && \
    curl -fsSL https://github.com/tamasfe/taplo/releases/latest/download/taplo-linux-x86_64.gz \
    | gzip -d - | install -m 755 /dev/stdin /usr/local/bin/taplo && taplo --version
