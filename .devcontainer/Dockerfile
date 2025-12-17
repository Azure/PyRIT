# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Makes installation faster
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

SHELL ["/bin/bash", "-c"]

USER root

# Install required system packages + ODBC prerequisites
RUN apt-get update && apt-get install -y \
    sudo \
    unixodbc \
    unixodbc-dev \
    libgl1-mesa-glx \
    git \
    curl \
    xdg-utils \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

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

ARG USERNAME=vscode
# Create the user and grant passwordless sudo
RUN groupadd $USERNAME \
    && useradd -g $USERNAME -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create venv using uv
RUN uv venv /opt/venv --python 3.11 \
 && chown -R vscode:vscode /opt/venv \
 && ls -la /opt/venv/bin/activate
ENV PATH="/opt/venv/bin:$PATH"

# Pre-create common user caches and fix permissions
RUN mkdir -p /home/vscode/.cache/pre-commit \
 && mkdir -p /home/vscode/.vscode-server \
 && mkdir -p /home/vscode/.cache/pip \
 && mkdir -p /home/vscode/.cache/uv \
 && mkdir -p /home/vscode/.cache/venv \
 && mkdir -p /home/vscode/.cache/pylance \
 && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.vscode-server \
 && chmod -R 777 /home/vscode/.cache/pip /home/vscode/.cache/pylance /home/vscode/.cache/venv /home/vscode/.cache/uv\
 && chmod -R 755 /home/vscode/.vscode-server

USER vscode
# Create bash configuration files and activate the venv in bash sessions
RUN touch /home/vscode/.bashrc /home/vscode/.bash_profile \
 && echo "[ -f /opt/venv/bin/activate ] && source /opt/venv/bin/activate" >> /home/vscode/.bashrc \
 && echo "[ -f /opt/venv/bin/activate ] && source /opt/venv/bin/activate" >> /home/vscode/.bash_profile

# Configure Git for better performance with bind mounts
RUN git config --global core.preloadindex true \
 && git config --global core.fscache true \
 && git config --global gc.auto 256 \
 && git config --global status.showUntrackedFiles all \
 && git config --global core.fsmonitor true

# Set uv’s cache directory so it can be mounted
ENV UV_CACHE_DIR="/home/vscode/.cache/uv"
