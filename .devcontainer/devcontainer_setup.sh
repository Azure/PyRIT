#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda activate pyrit-dev

if [ ! -d "/workspace/pyrit.egg-info" ]; then
    echo "Installing PyRIT dependencies..."
    cd /workspace
    pip install -e ".[dev,all]"
fi
