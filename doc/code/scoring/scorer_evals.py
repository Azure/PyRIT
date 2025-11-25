# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [markdown]
# # Scorer Evaluations - optional
# You may want to run evaluations on PyRIT scorers to see how their scores align with human assessment. This notebook demonstrates how to do that using a dataset of sample assistant responses and associated human labels. There are two types of scorer evaluations: those for `objective` datasets and those for `harm` datasets. Scoring for whether a response has achieved an objective is based on a true/false basis while scoring for harmfulness of a response (given a certain harm category) is based on a float scale (0.0 to 1.0 where 0.0 is minimal severity and 1.0 is maximal), often on a Likert basis. Metrics produced for `harm` scoring evaluation include mean absolute error, one-sample t-test statistics, and Krippendorff's alpha (for interrater agreement) while metrics produced for `objective` scoring evaluation include accuracy, F1 score, precision, and recall. More detailed information on each of the metrics can be found in the `scorer_evaluator` module [here](../../../pyrit/score/scorer_evaluation/scorer_evaluator.py).
#
# To actually run the evaluation in PyRIT, you need a `HumanLabeledDataset` object or a CSV file that includes columns of assistant responses, harm category or objectives (depending on the type of dataset you have), and human scores. You can then instantiate a PyRIT `Scorer` that you want to evaluate and pass it into a `HarmScorerEvaluator` or `ObjectiveScorerEvaluator` (which are subclasses of `ScorerEvaluator`). A `ScorerEvaluator` is responsible for running the end-to-end evaluation, and you can modify the number of model scoring trials as you see fit for your experiment. Note that for now, datasets with mixed harm categories cannot be used for evaluation, while datasets with mixed objectives can.
#
# You can also retrieve metrics from a specific evaluation or from an evaluation that was previously saved using the `get_scorer_metrics` method and passing in the name of the dataset (`dataset_name`) that the evaluation was run on.

# %%
# Imports:

from dataclasses import asdict
from pathlib import Path

from pyrit.common.path import SCORER_EVALS_HARM_PATH, SCORER_EVALS_TRUE_FALSE_PATH
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    LikertScalePaths,
    MetricsType,
    ObjectiveScorerEvaluator,
    ScorerEvaluator,
    SelfAskLikertScorer,
    SelfAskRefusalScorer,
    SelfAskTrueFalseScorer,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)
target = OpenAIChatTarget()

# %% [markdown]
# ## Running Harm Scorer Evaluation
# The following cell demonstrates how to run a custom harm scoring evaluation on a `Scorer` using a dataset of LLM assistant responses and human labels stored as a CSV file. In the following example, we use a small sample `hate_speech` dataset of assistant responses and human scores and run a harm evaluation using `SelfAskLikertScorer` with a Likert scale for hate speech. Sample datasets and saved metrics can be found in the `datasets/score/scorer_evals` directory.

# %% [markdown]
# #### Understanding Harm Scorer Metrics

# %% [markdown]
# When evaluating harm scorers, several metrics are computed to assess how well the model scores align with human evaluations:
#
# **Error Metrics:**
# - **`mean_absolute_error` (MAE)**: The average absolute difference between model scores and the mean of human scores (gold label). Lower values indicate better alignment. For example, an MAE of 0.15 means the model's scores differ from human scores by 0.15 on average (on a 0-1 scale).
# - **`mae_standard_error`**: The standard error of the MAE, used to calculate confidence intervals. Indicates the precision of the MAE estimate.
#
# **Statistical Significance:**
# - **`t_statistic`**: From a one-sample t-test comparing model scores to human scores. A high positive value suggests the model systematically scores higher than humans; a high negative value suggests it scores lower.
# - **`p_value`**: The probability of observing the difference between model and human scores by chance. Values < 0.05 typically indicate statistically significant differences.
#
# **Inter-Rater Reliability (Krippendorff's Alpha):**
#
# Krippendorff's alpha measures agreement between evaluators, ranging from -1.0 to 1.0:
# - **1.0**: Perfect agreement
# - **0.8-1.0**: Strong agreement
# - **0.6-0.8**: Moderate agreement
# - **0.0**: Agreement equivalent to chance
# - **< 0.0**: Systematic disagreement
#
# Three alpha values are reported:
# - **`krippendorff_alpha_humans`**: Agreement among human evaluators only. Measures consistency of the human-labeled "ground truth." Only computed when multiple human raters exist.
# - **`krippendorff_alpha_model`**: Agreement among multiple model scoring trials. Measures model consistency/reliability. Only computed when `num_scorer_trials > 1`.
# - **`krippendorff_alpha_combined`**: Agreement across all evaluators (humans + model trials combined). Measures how well the model aligns with the pool of human evaluators.

# %% [markdown]
# ### Multiple Evaluators:
#
# With multiple evaluators, we can measure inter-reliability alignment between evaluators shown below:

# %%
target = OpenAIChatTarget()
likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=LikertScalePaths.HATE_SPEECH_SCALE.value)

# factory method that creates an HarmScorerEvaluator in this case since metrics_type is HARM.
evaluator = ScorerEvaluator.from_scorer(scorer=likert_scorer, metrics_type=MetricsType.HARM)
csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/mini_hate_speech.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/hate_speech_multi_score.csv"

# The dataset_name defaults to "mini_hate_speech" but can be manually set to any string. Each response is graded once by the scorer.
metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
    csv_path=csv_path,
    assistant_response_col_name="assistant_response",
    human_label_col_names=["human_score_1", "human_score_2", "human_score_3"],
    objective_or_harm_col_name="category",
    num_scorer_trials=1,
    dataset_name="mini_hate_speech",
    assistant_response_data_type_col_name=None,
)

# Metrics are saved to datasets/score/scorer_evals/harm/results/mini_hate_speech_metrics.json
# Results from the model scoring trials are saved to datasets/score/scorer_evals/harm/results/mini_hate_speech_scoring_results.csv
asdict(metrics)

# %% [markdown]
# ### Single-Evaluators:
#
# The sample files below have only one human evaluator and thus the inter-reliability metric is not scored between human evaluators

# %%
from pyrit.score.scorer_evaluation.config_eval_datasets import get_harm_eval_datasets

harm_categories_to_evaluate = ["sexual_content"]

for harm_category in harm_categories_to_evaluate:
    harm_category_map = get_harm_eval_datasets(category=harm_category, metrics_type="harm")

    eval_rubric_path = harm_category_map["evaluation_rubric_file_path"]
    csv_path = str(Path(harm_category_map["dataset_file_path"]))

    likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=eval_rubric_path)

    evaluator = ScorerEvaluator.from_scorer(scorer=likert_scorer, metrics_type=MetricsType.HARM)

    # assistant_response_data_type_col_name is optional and can be used to specify the type of data for each response in the assistant response column.
    metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
        csv_path=csv_path,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["normalized_score_1"],
        objective_or_harm_col_name="category",
        num_scorer_trials=1,
        assistant_response_data_type_col_name=None,
        dataset_name=harm_category_map["dataset_name"],
    )

    print("Evaluation for harm category:", harm_category)
    print(asdict(metrics))

# %% [markdown]
# ## Retrieving Metrics
# You can retrieve the metrics from the above evaluation by calling the `get_scorer_metrics` from the `ScorerEvaluator` class or directly from the `Scorer` class and passing in the `dataset_name` (which in this case is `mini_hate_speech`). This will throw an error if evaluation has not yet been run on that dataset.

# %%
# Either work for fetching the hate_speech metrics
evaluator.get_scorer_metrics(dataset_name="mini_hate_speech")
likert_scorer.get_scorer_metrics(dataset_name="mini_hate_speech", metrics_type=MetricsType.HARM)

# Retrieve metrics for the full hate_speech dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="hate_speech")

# %% [markdown]
# ## Running Objective Scorer Evaluation
# The following cell demonstrates how to run a custom objective evaluation on a `Scorer` using a dataset of LLM assistant responses and human labels stored as a CSV file. This is much like the previous example except we use the `SelfAskRefusalScorer` that simply determines whether the model response was or was not a refusal.

# %% [markdown]
# #### Understanding Objective Scorer Metrics

# %% [markdown]
# When evaluating objective (true/false) scorers, the following metrics are computed based on the normalized score from humans as the gold label:
#
# - **`accuracy`**: The proportion of responses where the model's overall score matches the human overall score. Ranges from 0.0 to 1.0, where 1.0 means perfect agreement.
# - **`accuracy_standard_error`**: The standard error of the accuracy estimate, useful for constructing confidence intervals.
# - **`precision`**: Of all responses the model labeled as positive (True), what proportion were actually positive according to humans? High precision means few false positives.
# - **`recall`**: Of all responses that were actually positive according to humans, what proportion did the model correctly identify? High recall means few false negatives.
# - **`f1_score`**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance. Ranges from 0.0 to 1.0
#
# **Example Interpretation:**
# If a refusal scorer has accuracy=0.92, precision=0.95, recall=0.88, and f1_score=0.91, this means:
# - The model agrees with human normalized score 92% of the time
# - When the model says "this is a refusal," it's correct 95% of the time
# - The model catches 88% of actual refusals (missing 12%)
# - Overall performance is strong (F1=0.91)

# %%
refusal_scorer = SelfAskRefusalScorer(chat_target=target)

# factory method that creates an ObjectiveScorerEvaluator in this case because the refusal scorer is a true/false scorer.
evaluator = ScorerEvaluator.from_scorer(scorer=refusal_scorer)
csv_path = f"{str(SCORER_EVALS_TRUE_FALSE_PATH)}/mini_refusal.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_TRUE_FALSE_PATH)}/refusal.csv"

# assistant_response_data_type_col_name is optional and can be used to specify the type of data for each response in the assistant response column.
metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
    csv_path=csv_path,
    assistant_response_col_name="assistant_response",
    human_label_col_names=["normalized_score"],
    objective_or_harm_col_name="objective",
    assistant_response_data_type_col_name="data_type",
    num_scorer_trials=1,
)

refusal_scorer.get_scorer_metrics(dataset_name="mini_refusal")

# Retrieve metrics for the full refusal scorer dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="refusal")

# %% [markdown]
# In this case we use a `true_false_scorer` which determines whether the responses are harmful or not according to their harm category.

# %%
from pyrit.score.scorer_evaluation.config_eval_datasets import get_harm_eval_datasets

# set this list to the categories you want to evaluate
harm_categories_to_evaluate = ["information_integrity"]

for harm_category in harm_categories_to_evaluate:
    harm_category_map = get_harm_eval_datasets(category=harm_category, score_type="objective")
    eval_rubric_path = harm_category_map["evaluation_rubric_file_path"]
    csv_path = str(Path(harm_category_map["dataset_file_path"]))
    dataset_name = harm_category_map["dataset_name"]

    true_false_scorer = SelfAskTrueFalseScorer(true_false_question_path=Path(eval_rubric_path), chat_target=target)

    evaluator: ObjectiveScorerEvaluator = ScorerEvaluator.from_scorer(scorer=true_false_scorer)  # type: ignore

    metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
        csv_path=csv_path,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["normalized_score"],
        objective_or_harm_col_name="objective",
        assistant_response_data_type_col_name="data_type",
        num_scorer_trials=1,
    )

    print("Evaluation for harm category:", harm_category)
    print(asdict(metrics))
