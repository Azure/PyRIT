#!/usr/bin/env bash

# initializing conda in the current shell session
eval "$(conda shell.bash hook)"

conda create -n pyrit-dev python=3.11 -y
conda activate pyrit-dev
conda install ipykernel -y
pip install -e ."[dev,all]"
sudo chown -R vscode:vscode /workspace
