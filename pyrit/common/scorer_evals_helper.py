# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyrit.score.scorer import ScorerEvalConfig


def load_data_and_validate_cols(config: ScorerEvalConfig, tasks_accepted: bool) -> pd.DataFrame:
    eval_df = pd.read_csv(config.csv_path)
    required_columns = set(config.manual_grading_col_names + [config.assistant_response_col_name])
    if tasks_accepted and config.tasks_col_name:
        required_columns.add(config.tasks_col_name)
    assert required_columns.issubset(eval_df.columns), "Missing required columns in the dataset"
    num_responses = len(eval_df[config.assistant_response_col_name])
    for manual_col in config.manual_grading_col_names:
        assert (
            len(eval_df[manual_col]) == num_responses
        ), f"Number of scores in column {manual_col} does not match the number of responses"
    if config.tasks_col_name in required_columns:
        assert (
            len(eval_df[config.tasks_col_name]) == num_responses
        ), f"Number of tasks in column {config.tasks_col_name} does not match the number of responses"
    return eval_df


def extract_all_human_scores(eval_df: pd.DataFrame, config: ScorerEvalConfig) -> np.ndarray:
    all_human_scores = []
    for col in config.manual_grading_col_names:
        human_scores = eval_df[col].tolist()
        expected_len = len(eval_df[config.assistant_response_col_name])
        assert (
            len(human_scores) == expected_len
        ), f"Expected {expected_len} scores \
            in column {col} but got {len(human_scores)}"
        all_human_scores.append(human_scores)
    return np.array(all_human_scores)


def save_scores_to_csv(config, responses, all_model_scores, avg_model_scores, avg_human_scores, file_name):
    cols_dict = {"assistant_responses": responses}
    for trial, scores in enumerate(all_model_scores):
        cols_dict[f"trial {trial}"] = scores
    cols_dict["model_average"] = avg_model_scores
    cols_dict["human_average"] = avg_human_scores

    scores_df = pd.DataFrame(cols_dict)

    csv_path = Path(config.csv_scores_save_dir, f"{file_name}.csv")
    scores_df.to_csv(csv_path, index=False)


def save_metrics_json(config, eval_dict, file_name):
    json_path = Path(config.json_output_save_dir) / f"{file_name}.json"
    with open(json_path, "w") as f:
        json.dump(eval_dict, f, indent=4)
