FROM mcr.microsoft.com/devcontainers/miniconda:3

USER root

# Install required system packages
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    libgl1-mesa-glx \
    curl \
    xdg-utils \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install the DuckDB CLI
RUN curl -Lo duckdb.gz https://github.com/duckdb/duckdb/releases/download/v1.2.1/duckdb_cli-linux-amd64.gz \
 && gunzip duckdb.gz \
 && mv duckdb /usr/local/bin/duckdb \
 && chmod +x /usr/local/bin/duckdb

# Install the Azure CLI
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release \
 && curl -sL https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb -o packages-microsoft-prod.deb \
 && dpkg -i packages-microsoft-prod.deb \
 && rm packages-microsoft-prod.deb \
 && apt-get update \
 && apt-get install azure-cli -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda create -n pyrit-dev python=3.11 -y && \
    chown -R vscode:vscode /opt/conda/envs/pyrit-dev

RUN mkdir -p /home/vscode/.cache/pre-commit && \
    chown -R vscode:vscode /home/vscode/.cache/pre-commit

RUN mkdir -p /home/vscode/.vscode-server && \
    chown -R vscode:vscode /home/vscode/.vscode-server && \
    chmod -R 755 /home/vscode/.vscode-server

USER vscode

# Configure Git for better performance with bind mounts
RUN git config --global core.preloadindex true && \
    git config --global core.fscache true && \
    git config --global gc.auto 256 && \
    git config --global status.showUntrackedFiles all && \
    git config --global core.fsmonitor true

# Set pip’s cache directory so that we can mount it as a volume.
ENV PIP_CACHE_DIR="/home/vscode/.cache/pip"
