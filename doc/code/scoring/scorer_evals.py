# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scorer Evaluations
# This notebook demonstrates how to retrieve metrics for a Scorer, as well as how to run evaluations on a Scorer using a dataset of sample assistant responses and manual human scores. For now, this is only implemented for the `SelfAskLikertScorer`, specifically for the `hate_speech` and `violence` scales.

# %% [markdown]
# ## Retrieving metrics for a Scorer
# The following cell demonstrates how to retrieve baseline metrics for the `SelfAskLikertScorer` across all the scales that have been tested so far. These metrics were calculated based on datasets of prompts generated and manually scored by the AI Red Team. The manual scores--and thus the metrics--may **not** be reflective of general consensus of harmfulness.

# %%
import json

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.score import SelfAskLikertScorer

initialize_pyrit(memory_db_type=IN_MEMORY)

# Class method that returns a JSON string with the evaluation metrics across all scales tested so far
metrics = SelfAskLikertScorer.eval_stats_to_json()

# Calling this method on a scorer that doesn't have evaluation metrics yet will raise a ValueError
# metrics = SelfAskScaleScorer.eval_stats_to_json()

# Load metrics as a Python dict: includes mean absolute error (MAE), standard error of MAE, t-statistic and p-value (from 1-sample t-test using
# model scores - human scores), and Krippendorff's Alpha for inter-rater reliability across all model and human scores
metrics_dict = json.loads(metrics)

metrics_dict


# %% [markdown]
# ## Running evaluation on a Scorer
# The following cell demonstrates how to run a custom evaluation on a `Scorer` using a dataset of LLM assistant responses and human-labeled scores. Use a `ScorerEvalConfig` object to configure your evaluation. In the following example, we use a small sample `hate_speech` dataset of assistant responses and human scores to demonstrate configuration and running of the evaluation on the `SelfAskLikertScorer` using the PyRIT `HATE_SPEECH_SCALE`.

# %%
import json
import pathlib

from pyrit.common.path import (
    SCORER_EVALS_PATH,
    SCORER_EVALS_RESULTS_METRICS_PATH,
    SCORER_EVALS_RESULTS_SCORES_CSV_PATH,
)
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import LikertScalePaths, ScorerEvalConfig, SelfAskLikertScorer

target = OpenAIChatTarget()
likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=LikertScalePaths.HATE_SPEECH_SCALE.value)

# ScorerEvalConfig object can either be manually instantiated or loaded from a yaml file.
config_path = pathlib.Path(SCORER_EVALS_PATH, "eval_configs", "sample_config_likert_hate.yaml").resolve()
config = ScorerEvalConfig.from_yaml_file(config_path)

# The below code is equivalent to the above but manually instantiates the ScorerEvalConfig object.
config = ScorerEvalConfig(
    csv_path=f"{str(SCORER_EVALS_PATH)}/eval_datasets/likert_hate_speech_sample_dataset.csv",
    assistant_response_col_name="assistant_response",
    manual_grading_col_names=["human_likert_score_1", "human_likert_score_2", "human_likert_score_3"],
    normalized=False,
    scorer_trials=1,
    tasks_col_name=None,
    json_output_save_dir=str(SCORER_EVALS_RESULTS_METRICS_PATH),
    csv_scores_save_dir=str(SCORER_EVALS_RESULTS_SCORES_CSV_PATH),
)
# When the evaluation is run, the metrics as a JSON file and Scorer scores as a .csv file are saved inside the scorer_evals_results
# directory by default. This directory is created if it doesn't exist.
score_dict = await likert_scorer.run_evaluation(config=config)  # type: ignore

# Load the metrics as a Python dict
json.loads(score_dict)
