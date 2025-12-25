# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pathlib import Path

from pyrit.common.path import (
    SCORER_EVALS_LIKERT_PATH,
    SCORER_EVALS_TRUE_FALSE_PATH,
    SCORER_LIKERT_PATH,
)
from pyrit.score.true_false.self_ask_true_false_scorer import TRUE_FALSE_QUESTIONS_PATH


def get_harm_eval_datasets(category: str, metrics_type: str):
    """
    Get the configuration for harm evaluation datasets based on category and metrics type.

    Args:
        category (str): The harm category (e.g., "exploit", "hate_speech", etc.).
        metrics_type (str): The type of metrics ("harm" or "objective").

    Returns:
        dict: A dictionary containing dataset file path, evaluation rubric file path, and dataset name.

    Raises:
        ValueError: If an unknown metrics type is provided or if the category is not found.
    """
    directory = ""
    if metrics_type == "harm":
        directory = str(SCORER_EVALS_LIKERT_PATH)
        eval_directory = SCORER_LIKERT_PATH
    elif metrics_type == "objective":
        directory = str(SCORER_EVALS_OBJECTIVE_PATH)
        eval_directory = TRUE_FALSE_QUESTIONS_PATH

    else:
        raise ValueError(f"Unknown metrics type: {metrics_type}")

    # Update dataset file paths with custom files/setup if needed and/or additional harm categories
    # Map harm categories to respective dataset file paths and evaluation rubric file paths
    # This allows for easy extension to new harm categories by editing entries in map.
    # Also allows for easy editing of dataset file paths and eval rubric file paths without changing code logic.

    harm_category_map = {
        "exploit": {
            "dataset_file_path": f"{directory}/exploit.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "exploits.yaml")),
            "dataset_name": "exploit_dataset",
        },
        "hate_speech": {
            "dataset_file_path": f"{directory}/hate_speech.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "hate_speech.yaml")),
            "dataset_name": "hate_speech_dataset",
        },
        "information_integrity": {
            "dataset_file_path": f"{directory}/info_integrity.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "information_integrity.yaml")),
            "dataset_name": "information_integrity_dataset",
        },
        "privacy": {
            "dataset_file_path": f"{directory}/privacy.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "privacy.yaml")),
            "dataset_name": "privacy_dataset",
        },
        "self-harm": {
            "dataset_file_path": f"{directory}/self_harm.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "self_harm.yaml")),
            "dataset_name": "self_harm_dataset",
        },
        "sexual_content": {
            "dataset_file_path": f"{directory}/sexual.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "sexual.yaml")),
            "dataset_name": "sexual_content_dataset",
        },
        "violence": {
            "dataset_file_path": f"{directory}/violence.csv",
            "evaluation_rubric_file_path": str(Path(eval_directory, "violence.yaml")),
            "dataset_name": "violence_dataset",
        },
    }

    if category not in harm_category_map:
        raise ValueError(
            f"Harm category '{category}' not found in harm_category_map. \n"
            f"Please add it to the map with the appropriate dataset and rubric file paths."
        )

    return harm_category_map[category]
