# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Scorer Evaluations
# You may want to run evaluations on PyRIT scorers to see how their scores align with human assessment. This notebook demonstrates how to do that using a dataset of sample assistant responses and associated human labels. There are two types of scorer evaluations: those for `objective` datasets and those for `harm` datasets. Scoring for whether a response has achieved an objective is based on a true/false basis while scoring for harmfulness of a response (given a certain harm category) is based on a float scale (0.0 to 1.0 where 0.0 is minimal severity and 1.0 is maximal), often on a Likert basis. Metrics produced for `harm` scoring evaluation include mean absolute error, one-sample t-test statistics, and Krippendorff's alpha (for interrater agreement) while metrics produced for `objective` scoring evaluation include accuracy, F1 score, precision, and recall.
#
# To actually run the evaluation in PyRIT, you need a `HumanLabeledDataset` object or a CSV file that includes columns of assistant responses, harm category or objectives (depending on the type of dataset you have), and human scores. You can then instantiate a PyRIT `Scorer` that you want to evaluate and pass it into a `HarmScorerEvaluator` or `ObjectiveScorerEvaluator` (which are subclasses of `ScoreEvaluator`). A `ScorerEvaluator` is responsible for running the end-to-end evaluation, and you can modify the number of model scoring trials as you see fit for your experiment. Note that for now, datasets with mixed harm categories cannot be used for evaluation, while datasets with mixed objectives can.
#
# You can also retrieve metrics from a specific evaluation or from an evaluation that was previously saved using the `get_scorer_metrics` method and passing in the `dataset_name`.

# %% [markdown]
# ## Running Harm Scorer Evaluation
# The following cell demonstrates how to run a custom harm evaluation on a `Scorer` using a dataset of LLM assistant responses and human labels stored as a CSV file. In the following example, we use a small sample `hate_speech` dataset of assistant responses and human scores and run a harm evaluation using `SelfAskLikertScorer` with a Likert scale for hate speech. Sample datasets and saved metrics can be found in the `datasets/score/scorer_evals` directory.

# %%
from dataclasses import asdict

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.path import SCORER_EVALS_HARM_PATH
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import LikertScalePaths, ScorerEvaluator, SelfAskLikertScorer

initialize_pyrit(memory_db_type=IN_MEMORY)

target = OpenAIChatTarget()
likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=LikertScalePaths.HATE_SPEECH_SCALE.value)

# factory method that creates an HarmScorerEvaluator in this case
evaluator = ScorerEvaluator.from_scorer(scorer=likert_scorer, metrics_type="harm")
csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/harm/SAMPLE_hate_speech.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/hate_speech.csv"

# The dataset_name defaults to "SAMPLE_hate_speech" but can be manually set to any string. Each response is graded once by the scorer.
metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
    csv_path=csv_path,
    assistant_response_col="assistant_response",
    human_label_col_names=["human_likert_score_1", "human_likert_score_2", "human_likert_score_3"],
    objective_or_harm_col_name="category",
    num_scorer_trials=1,
)

# Metrics are saved to datasets/score/scorer_evals/harm/SAMPLE_hate_speech_SelfAskLikertScorer_metrics.json
# Results from the model scoring trials are saved to datasets/score/scorer_evals/harm/SAMPLE_hate_speech_SelfAskLikertScorer_scoring_results.csv
asdict(metrics)

# %% [markdown]
# ## Retrieving Metrics
# You can retrieve the metrics from the above evaluation by calling the `get_scorer_metrics` from the `ScorerEvaluator` class or directly from the `Scorer` class and passing in the `dataset_name` (which in this case is `SAMPLE_hate_speech`). This will throw an error if evaluation has not yet been run on that dataset.

# %%
# Either work for fetching the hate_speech metrics
evaluator.get_scorer_metrics(dataset_name="SAMPLE_hate_speech")
likert_scorer.get_scorer_metrics(dataset_name="SAMPLE_hate_speech", metrics_type="harm")

# Retrieve metrics for the full hate_speech dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="hate_speech")

# %% [markdown]
# ## Running Objective Scorer Evaluation
# The following cell demonstrates how to run a custom objective evaluation on a `Scorer` using a dataset of LLM assistant responses and human labels stored as a CSV file. This is much like the previous example except we use the `SelfAskRefusalScorer` that simply determines whether the model response was or was not a refusal.

# %%
from pyrit.common.path import SCORER_EVALS_OBJECTIVE_PATH
from pyrit.score import SelfAskRefusalScorer

target = OpenAIChatTarget()
refusal_scorer = SelfAskRefusalScorer(chat_target=target)

# factory method that creates an ObjectiveScorerEvaluator in this case
evaluator = ScorerEvaluator.from_scorer(scorer=refusal_scorer)
csv_path = f"{str(SCORER_EVALS_OBJECTIVE_PATH)}/SAMPLE_mixed_objective_refusal.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_OBJECTIVE_PATH)}/mixed_objective_refusal.csv"

metrics = await evaluator.run_evaluation_from_csv_async(  # type:ignore
    csv_path=csv_path,
    assistant_response_col="assistant_message",
    human_label_col_names=["human_score"],
    objective_or_harm_col_name="objective",
    num_scorer_trials=1,
)

refusal_scorer.get_scorer_metrics(dataset_name="SAMPLE_mixed_objective_refusal")

# Retrieve metrics for the full refusal scorer dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="mixed_objective_refusal")
