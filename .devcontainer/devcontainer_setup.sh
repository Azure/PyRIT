#!/bin/bash
set -e

MYPY_CACHE="/workspace/.mypy_cache"
# Create the mypy cache directory if it doesn't exist
if [ ! -d "$MYPY_CACHE" ]; then
    echo "Creating mypy cache directory..."
    sudo mkdir -p $MYPY_CACHE
    sudo chown vscode:vscode $MYPY_CACHE
    sudo chmod 777 $MYPY_CACHE
else
    # Check ownership
    OWNER=$(stat -c '%U:%G' $MYPY_CACHE)

    if [ "$OWNER" != "vscode:vscode" ]; then
        echo "Fixing mypy cache directory ownership..."
        sudo chown -R vscode:vscode $MYPY_CACHE
    fi

    # Check permissions
    PERMS=$(stat -c '%a' $MYPY_CACHE)

    if [ "$PERMS" != "777" ]; then
        echo "Fixing mypy cache directory permissions..."
        sudo chmod -R 777 $MYPY_CACHE
    fi
fi

# cleanup old extensions
rm -rf /home/vscode/.vscode-server/extensions/{*,.[!.]*,..?*}

# Path to store the hash
HASH_FILE="/home/vscode/.cache/pip/pyproject_hash"

# Make sure the hash file is writable if it exists; if not, it will be created
if [ -f "$HASH_FILE" ]; then
    chmod 666 "$HASH_FILE"
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate pyrit-dev

# Compute current hash
CURRENT_HASH=$(sha256sum /workspace/pyproject.toml | awk '{print $1}')

# Check if hash file exists and if the hash has changed
if [ ! -f "$HASH_FILE" ] || [ "$(cat $HASH_FILE)" != "$CURRENT_HASH" ]; then
    echo "📦 pyproject.toml has changed, installing environment..."

    # Install dependencies
    conda install ipykernel -y
    pip install -e '.[dev,all]'

    # Save the new hash
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "✅ pyproject.toml has not changed, skipping installation."
fi

echo "🚀 Dev container setup complete!"
