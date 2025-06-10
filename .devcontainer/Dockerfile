FROM mcr.microsoft.com/devcontainers/anaconda:3

SHELL ["/bin/bash", "-c"]

USER root

# Install required system packages + ODBC prerequisites
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

# Install the Azure CLI, Microsoft ODBC Driver 18 & SQL tools
RUN apt-get update && apt-get install -y \
      apt-transport-https \
      ca-certificates \
      gnupg \
      lsb-release \
 && curl -sL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
      -o packages-microsoft-prod.deb \
 && dpkg -i packages-microsoft-prod.deb \
 && rm packages-microsoft-prod.deb \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y \
      msodbcsql18 \
      mssql-tools \
      unixodbc-dev \
 && apt-get install -y azure-cli \
 && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> /etc/profile.d/sqltools.sh \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# audio back-ends needed by Azure Speech SDK
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 apt-get install -y --no-install-recommends \
      libasound2 \
      libpulse0 \
 && rm -rf /var/lib/apt/lists/*

# Create conda env and install pyodbc into it
RUN conda create -n pyrit-dev python=3.11 -y && \
    conda install -n pyrit-dev -c conda-forge pyodbc -y && \
    chown -R vscode:vscode /opt/conda/envs/pyrit-dev

# Pre-create common user caches and fix permissions
RUN mkdir -p /home/vscode/.cache/pre-commit \
 && mkdir -p /home/vscode/.vscode-server \
 && mkdir -p /home/vscode/.cache/pip \
 && mkdir -p /home/vscode/.cache/conda \
 && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.vscode-server \
 && chmod -R 777 /home/vscode/.cache/conda /home/vscode/.cache/pip \
 && chmod -R 755 /home/vscode/.vscode-server

USER vscode

# Configure Git for better performance with bind mounts
RUN git config --global core.preloadindex true \
 && git config --global core.fscache true \
 && git config --global gc.auto 256 \
 && git config --global status.showUntrackedFiles all \
 && git config --global core.fsmonitor true

# Set pip’s cache directory so it can be mounted
ENV PIP_CACHE_DIR="/home/vscode/.cache/pip"
