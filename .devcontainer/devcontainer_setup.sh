#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda activate pyrit-dev

if [ ! -d "/workspace/pyrit.egg-info" ]; then
    echo "Installing PyRIT dependencies..."
    cd /workspace
    conda install ipykernel -y
    pip install -e ".[dev,all]"
fi
