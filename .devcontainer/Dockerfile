FROM mcr.microsoft.com/devcontainers/anaconda:3

RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    libgl1-mesa-glx \ 
    curl \
    xdg-utils

# Download and install the DuckDB CLI.
# Change this line if a newer version is available.
RUN curl -Lo duckdb.gz https://github.com/duckdb/duckdb/releases/download/v1.2.1/duckdb_cli-linux-amd64.gz && \
    gunzip duckdb.gz && \
    mv duckdb /usr/local/bin/duckdb && \
    chmod +x /usr/local/bin/duckdb

# Install Azure CLI (Debian variant)
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
    curl -sL https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb -o packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install azure-cli -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*