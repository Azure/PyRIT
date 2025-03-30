#!/bin/bash
set -e

# Clone PyRIT repository if not already present
if [ ! -d "/app/PyRIT" ]; then
    echo "Cloning PyRIT repository..."
    git clone https://github.com/Azure/PyRIT
else
    echo "PyRIT repository already exists. Updating..."
    cd /app/PyRIT
    git pull
    cd /app
fi

# Copy doc folder to notebooks directory
echo "Copying documentation to notebooks directory..."
cp -r /app/PyRIT/doc/* /app/notebooks/
rm -rf /app/PyRIT

# Default to CPU mode
export CUDA_VISIBLE_DEVICES="-1"

# Only try to use GPU if explicitly enabled
if [ "$ENABLE_GPU" = "true" ] && command -v nvidia-smi &> /dev/null; then
    echo "GPU detected and explicitly enabled, running with GPU support"
    export CUDA_VISIBLE_DEVICES="0"
else
    echo "Running in CPU-only mode"
    export CUDA_VISIBLE_DEVICES="-1"
fi

# Print PyRIT version
python -c "import pyrit; print(f'Running PyRIT version: {pyrit.__version__}')"

# Execute the command passed to docker run (or the CMD if none provided)
exec "$@"
