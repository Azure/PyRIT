#!/bin/bash
set -e

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
