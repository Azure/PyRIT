#!/bin/bash

export LOG=$1
export model=$2 # llama2 or mistral

python -u ../evaluate_individual.py \
    --config="../configs/transfer_${model}.py" \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="${LOG}" \
    --config.n_train_data=2 \
    --config.n_test_data=0
