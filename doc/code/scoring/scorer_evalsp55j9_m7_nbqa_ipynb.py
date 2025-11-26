# %%NBQA-CELL-SEP52c935
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

# %%NBQA-CELL-SEP52c935
target = OpenAIChatTarget()
likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=LikertScalePaths.HATE_SPEECH_SCALE.value)

# factory method that creates an HarmScorerEvaluator in this case since metrics_type is HARM.
evaluator = ScorerEvaluator.from_scorer(scorer=likert_scorer, metrics_type=MetricsType.HARM)
csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/mini_hate_speech.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/hate_speech_multi_score.csv"

# The dataset_name defaults to "mini_hate_speech" but can be manually set to any string. Each response is graded once by the scorer.
metrics = await evaluator.run_evaluation_from_csv_async( #type:ignore
    csv_path=csv_path, 
    assistant_response_col_name="assistant_response", 
    human_label_col_names=["human_score_1", "human_score_2", "human_score_3"], 
    objective_or_harm_col_name= "category", 
    num_scorer_trials=1,
    dataset_name="mini_hate_speech",
    assistant_response_data_type_col_name=None,
)

# Metrics are saved to datasets/score/scorer_evals/harm/results/mini_hate_speech_metrics.json
# Results from the model scoring trials are saved to datasets/score/scorer_evals/harm/results/mini_hate_speech_scoring_results.csv
asdict(metrics)

# %%NBQA-CELL-SEP52c935
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
        dataset_name = harm_category_map["dataset_name"],
        
    )

    print("Evaluation for harm category:", harm_category)
    print(asdict(metrics))

# %%NBQA-CELL-SEP52c935
# Either work for fetching the hate_speech metrics
evaluator.get_scorer_metrics(dataset_name = "mini_hate_speech")
likert_scorer.get_scorer_metrics(dataset_name = "mini_hate_speech", metrics_type=MetricsType.HARM)

# Retrieve metrics for the full hate_speech dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="hate_speech")

# %%NBQA-CELL-SEP52c935
refusal_scorer = SelfAskRefusalScorer(chat_target=target)

# factory method that creates an ObjectiveScorerEvaluator in this case because the refusal scorer is a true/false scorer.
evaluator = ScorerEvaluator.from_scorer(scorer=refusal_scorer)
csv_path = f"{str(SCORER_EVALS_TRUE_FALSE_PATH)}/mini_refusal.csv"
# Uncomment the line below to use the full dataset of approx 200 entries
# csv_path = f"{str(SCORER_EVALS_TRUE_FALSE_PATH)}/refusal.csv"

# assistant_response_data_type_col_name is optional and can be used to specify the type of data for each response in the assistant response column.
metrics = await evaluator.run_evaluation_from_csv_async( #type:ignore
    csv_path=csv_path, 
    assistant_response_col_name="assistant_response", 
    human_label_col_names=["normalized_score"], 
    objective_or_harm_col_name= "objective",
    assistant_response_data_type_col_name="data_type", 
    num_scorer_trials=1,
)

refusal_scorer.get_scorer_metrics(dataset_name="mini_refusal")

# Retrieve metrics for the full refusal scorer dataset that have already been computed and saved by the PyRIT team.
# full_metrics = likert_scorer.get_scorer_metrics(dataset_name="refusal")

# %%NBQA-CELL-SEP52c935
from pyrit.score.scorer_evaluation.config_eval_datasets import get_harm_eval_datasets

# set this list to the categories you want to evaluate
harm_categories_to_evaluate = ["information_integrity"]

for harm_category in harm_categories_to_evaluate:
    harm_category_map = get_harm_eval_datasets(category=harm_category, metrics_type="objective")
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
