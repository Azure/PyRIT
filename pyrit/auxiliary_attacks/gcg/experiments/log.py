# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import mlflow
import subprocess as sp


def log_params(params, param_keys=["model_name", "transfer", "n_train_data", "n_test_data", "n_steps", "batch_size"]):
    mlflow_params = {key: params.to_dict()[key] for key in param_keys}
    mlflow.log_params(mlflow_params)


def log_train_goals(train_goals):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    train_goals_str = "\n".join(train_goals)
    mlflow.log_text(train_goals_str, f"train_goals_{timestamp}.txt")


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = {f"gpu{i+1}_free_memory": int(val.split()[0]) for i, val in enumerate(memory_free_info)}
    memory_free_string = ", ".join(f"{val} MiB" for val in memory_free_values.values())
    print(f"Free GPU memory:\n{memory_free_string}")
    return memory_free_values


def log_gpu_memory(step, synchronous=False):
    memory_values = get_gpu_memory()
    for gpu, val in memory_values.items():
        mlflow.log_metric(gpu, val, step=step, synchronous=synchronous)


def log_loss(step, loss, synchronous=False):
    mlflow.log_metric("loss", loss, step=step, synchronous=synchronous)


def log_table_summary(losses, controls, n_steps):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mlflow.log_table(
        {
            "step": [i + 1 for i in range(n_steps)],
            "loss": losses,
            "control": controls,
        },
        artifact_file=f"gcg_results_{timestamp}.json",
    )
