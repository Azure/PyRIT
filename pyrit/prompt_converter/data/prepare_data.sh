#!/bin/bash

set -e

if ! command -v unzip &> /dev/null
then
    echo "'unzip' command not found, please install zip to prepare data"
    exit
fi
# create sample problematic example data
wget https://maartensap.com/social-bias-frames/SBIC.v2.tgz -P sbf/
tar -zxf sbf/SBIC.v2.tgz --directory sbf
python gen_sample_utterances.py

# download data which will be sampled for few-shot
wget https://huggingface.co/datasets/ariesutiono/entailment-bank-v3/raw/main/task1_train.jsonl -P entailmentbank
wget https://github.com/nyu-mll/nope/raw/main/annotated_corpus/nope-v1.zip
unzip nope-v1.zip
mv corpus_package nope
git clone -b imppres --single-branch git@github.com:alexwarstadt/data_generation.git data_generation_imppres

# process data for few-shot
cd ../
python exemplars.py
