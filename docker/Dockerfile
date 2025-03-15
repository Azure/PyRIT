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

# Copy requirements file
COPY requirements.txt /app/

# Install all Python dependencies at once with pinned versions
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Install PyRIT from PyPI (the official way)
RUN pip install --no-cache-dir pyrit[dev,all]

# Create a directory for notebooks and data
RUN mkdir -p /app/notebooks /app/data && \
    chmod -R 777 /app/notebooks /app/data

# Check PyRIT version
RUN python -c "import pyrit; print(f'PyRIT version: {pyrit.__version__}')"

# Create non-root user for better security
RUN useradd -ms /bin/bash pyrituser
RUN chown -R pyrituser:pyrituser /app/notebooks /app/data

# Create and set permissions for the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
RUN chown pyrituser:pyrituser /app/start.sh

# Switch to non-root user
USER pyrituser

# Expose port for JupyterLab
EXPOSE 8888

# Set the entrypoint to the startup script
ENTRYPOINT ["/app/start.sh"]