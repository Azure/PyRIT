# Base image from Microsoft Azure with Python 3.12
FROM mcr.microsoft.com/azureml/minimal-py312-inference:20250310.v1

LABEL description="Docker container for PyRIT with Jupyter Notebook integration"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Switch to root user to install packages
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    unixodbc \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install JupyterLab and related packages
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    pandas \
    seaborn

# Install PyRIT from PyPI (the official way)
RUN pip install --no-cache-dir pyrit[dev,all]

# Create a directory for notebooks and data
RUN mkdir -p /app/notebooks /app/data && \
    chmod -R 777 /app/notebooks /app/data

# Check PyRIT version
RUN python -c "import pyrit; print(f'PyRIT version: {pyrit.__version__}')"

# Expose port for JupyterLab
EXPOSE 8888

# Create and set permissions for the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set the entrypoint to the startup script
ENTRYPOINT ["/app/start.sh"]