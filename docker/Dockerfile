# Base image from Microsoft Azure with Python 3.12
FROM mcr.microsoft.com/azureml/minimal-py312-inference:20250310.v1
LABEL description="Docker container for PyRIT with Jupyter Notebook and GUI support"

# Build arguments for version tracking
ARG PYRIT_SOURCE=pypi
ARG PYRIT_VERSION=""
ARG GIT_COMMIT=""
ARG GIT_MODIFIED=false

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV JUPYTER_ENABLE_LAB=yes
ENV CUDA_VISIBLE_DEVICES=-1
ENV ENABLE_GPU=false

# Switch to root user to install packages
USER root

# Install system dependencies and Node.js 20.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    unixodbc \
    libgl1-mesa-glx \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# For PyPI: just install the package
RUN if [ "$PYRIT_SOURCE" = "pypi" ]; then \
        echo "Installing PyRIT from PyPI version: $PYRIT_VERSION"; \
        pip install --no-cache-dir pyrit[dev,speech,opencv,fairness_bias,fastapi,playwright]==$PYRIT_VERSION; \
    fi

# For local: copy source, install editable, and build frontend
COPY --chown=dockeruser:dockeruser pyproject.toml MANIFEST.in README.md LICENSE /app/
COPY --chown=dockeruser:dockeruser pyrit/ /app/pyrit/
COPY --chown=dockeruser:dockeruser frontend/ /app/frontend/
COPY --chown=dockeruser:dockeruser build_scripts/ /app/build_scripts/
COPY --chown=dockeruser:dockeruser doc/ /app/doc/

RUN if [ "$PYRIT_SOURCE" = "local" ]; then \
        echo "Installing PyRIT from local source"; \
        pip install --no-cache-dir -e .[dev,speech,opencv,fairness_bias,fastapi,playwright]; \
        echo "Building frontend..."; \
        python build_scripts/prepare_package.py; \
    fi

# Create build info file
RUN python -c "import json; import pyrit; \
info = { \
    'source': '$PYRIT_SOURCE', \
    'version': pyrit.__version__, \
    'commit': '$GIT_COMMIT' if '$GIT_COMMIT' else None, \
    'modified': '$GIT_MODIFIED' == 'true', \
    'display': '$PYRIT_VERSION' if '$PYRIT_SOURCE' == 'pypi' else ('$GIT_COMMIT' + (' + local changes' if '$GIT_MODIFIED' == 'true' else '') if '$GIT_COMMIT' else pyrit.__version__) \
}; \
f = open('/app/build_info.json', 'w'); json.dump(info, f); f.close()"


# Create a directory for notebooks and data
RUN mkdir -p /app/notebooks /app/data /app/assets && \
    chmod -R 777 /app/notebooks /app/data /app/assets

# Copy doc to notebooks for Jupyter mode
RUN if [ -d "/app/doc" ]; then \
        cp -r /app/doc/* /app/notebooks/ || true; \
    fi

# Check PyRIT version
RUN python -c "import pyrit; print(f'PyRIT version: {pyrit.__version__}')"

RUN chown -R dockeruser:dockeruser /app

# Create and set permissions for the startup script
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to non-root user
USER dockeruser

# Expose ports for JupyterLab (8888) and GUI (8000)
EXPOSE 8888 8000

# Set the entrypoint to the startup script (mode determined by PYRIT_MODE env var)
ENTRYPOINT ["/app/start.sh"]
