# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import numpy as np
from scipy.stats import ttest_1samp

from pyrit.common.utils import verify_and_resolve_path
from pyrit.score import Scorer
from pyrit.score.scorer_evaluation.human_labeled_dataset import (
    HarmHumanLabeledEntry,
    HumanLabeledDataset,
    ObjectiveHumanLabeledEntry,
)
from pyrit.score.scorer_evaluation.metrics_type import MetricsType
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

from .krippendorff import krippendorff_alpha

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ScorerMetrics")

# Standard column names for evaluation datasets
STANDARD_HUMAN_LABEL_COL = "human_score"
STANDARD_OBJECTIVE_COL = "objective"
STANDARD_HARM_COL = "harm_category"
STANDARD_ASSISTANT_RESPONSE_COL = "assistant_response"


@dataclass
class ScorerEvalDatasetFiles:
    """
    Configuration for evaluating a scorer on a set of dataset files.
    
    Maps input dataset files (via glob patterns) to an output result file.
    Multiple files matching the patterns will be concatenated before evaluation.
    
    Args:
        human_labeled_datasets_files (List[str]): List of glob patterns to match CSV files.
            Examples: ["objective/*.csv"], ["objective/hate_speech.csv", "objective/violence.csv"]
        result_file (str): Name of the result file (stem used as dict key in results).
            Example: "objective_evaluation_results.jsonl"
    """
    human_labeled_datasets_files: List[str]
    result_file: str


@dataclass
class ScorerMetrics:
    """
    Base dataclass for storing scorer evaluation metrics.

    This class provides methods for serializing metrics to JSON and loading them from JSON files.
    """

    def to_json(self) -> str:
        """
        Convert the metrics to a JSON string.

        Returns:
            str: The JSON string representation of the metrics.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls: Type[T], file_path: Union[str, Path]) -> T:
        """
        Load the metrics from a JSON file.

        Args:
            file_path (Union[str, Path]): The path to the JSON file.

        Returns:
            ScorerMetrics: An instance of ScorerMetrics with the loaded data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        file_path = verify_and_resolve_path(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class HarmScorerMetrics(ScorerMetrics):
    """
    Metrics for evaluating a harm scorer against a HumanLabeledDataset.

    Args:
        mean_absolute_error (float): The mean absolute error between the model scores and the gold scores.
        mae_standard_error (float): The standard error of the mean absolute error. This can be used to calculate
            a confidence interval for the mean absolute error.
        t_statistic (float): The t-statistic for the one-sample t-test comparing model scores to human scores with a
            null hypothesis that the mean difference is 0. A high positive t-statistic (along with a low p-value)
            indicates that the model scores are typically higher than the human scores.
        p_value (float): The p-value for the one-sample t-test above. It represents the probability of obtaining a
            difference in means as extreme as the observed difference, assuming the null hypothesis is true.
        krippendorff_alpha_combined (float): Krippendorff's alpha for the reliability data, which includes both
            human and model scores. This measures the agreement between all the human raters and model scoring trials
            and ranges between -1.0 to 1.0 where 1.0 indicates perfect agreement, 0.0 indicates no agreement, and
            negative values indicate systematic disagreement.
        krippendorff_alpha_humans (float, Optional): Krippendorff's alpha for human scores, if there are
            multiple human raters. This measures the agreement between human raters.
        krippendorff_alpha_model (float, Optional): Krippendorff's alpha for model scores, if there are
            multiple model scoring trials. This measures the agreement between model scoring trials.
    """

    mean_absolute_error: float
    mae_standard_error: float
    t_statistic: float
    p_value: float
    krippendorff_alpha_combined: float
    krippendorff_alpha_humans: Optional[float] = None
    krippendorff_alpha_model: Optional[float] = None
    trial_scores: Optional[np.ndarray] = None


@dataclass
class ObjectiveScorerMetrics(ScorerMetrics):
    """
    Metrics for evaluating an objective scorer against a HumanLabeledDataset.

    Args:
        accuracy (float): The accuracy of the model scores when using the majority vote of
            human scores as the gold label.
        f1_score (float): The F1 score of the model scores, an indicator of performance of the
            LLM scorer in its alignment with human scores.
        precision (float): The precision of the model scores, an indicator of the model's accuracy
            in its positive predictions.
        recall (float): The recall of the model scores, an indicator of the model's ability to correctly
            identify positive labels.
        trial_scores (Optional[np.ndarray]): The raw scores from each trial. Shape is (num_trials, num_responses).
            Useful for debugging and analyzing scorer variance.
    """

    accuracy: float
    accuracy_standard_error: float
    f1_score: float
    precision: float
    recall: float
    trial_scores: Optional[np.ndarray] = None


class ScorerEvaluator(abc.ABC):
    """
    A class that evaluates an LLM scorer against HumanLabeledDatasets, calculating appropriate
    metrics and saving them to a file.
    """

    def __init__(self, scorer: Scorer):
        """
        Initialize the ScorerEvaluator with a scorer.

        Args:
            scorer (Scorer): The scorer to evaluate.
        """
        self.scorer = scorer

    @classmethod
    def from_scorer(cls, scorer: Scorer, metrics_type: Optional[MetricsType] = None) -> "ScorerEvaluator":
        """
        Create a ScorerEvaluator based on the type of scoring.

        Args:
            scorer (Scorer): The scorer to evaluate.
            metrics_type (MetricsType): The type of scoring, either HARM or OBJECTIVE.
                If not provided, it will default to OBJECTIVE for true/false scorers and HARM for all other
                scorers.

        Returns:
            ScorerEvaluator: An instance of HarmScorerEvaluator or ObjectiveScorerEvaluator.
        """
        if not metrics_type:
            metrics_type = MetricsType.OBJECTIVE if isinstance(scorer, TrueFalseScorer) else MetricsType.HARM

        _EVALUATOR_MAP = {MetricsType.HARM: HarmScorerEvaluator, MetricsType.OBJECTIVE: ObjectiveScorerEvaluator}

        evaluator = _EVALUATOR_MAP.get(metrics_type, HarmScorerEvaluator)
        return evaluator(scorer=scorer)

    async def run_evaluation_from_files_async(
        self,
        *,
        dataset_files: List[ScorerEvalDatasetFiles],
        num_scorer_trials: int = 1,
        add_to_registry: bool = False,
    ) -> dict[str, ScorerMetrics]:
        """
        Evaluate scorer using dataset files configuration.
        
        For each dataset files entry:
        - Collect all files matching the glob patterns
        - Load each file as a HumanLabeledDataset
        - Combine the datasets
        - Evaluate the combined dataset
        - Return metrics keyed by result file stem
        
        Args:
            dataset_files: List of ScorerEvalDatasetFiles configurations.
                Each specifies glob patterns for input files and a result file name.
            num_scorer_trials: Number of scoring trials per response. Defaults to 1.
            add_to_registry: Whether to save to official results. Defaults to False.
        
        Returns:
            Dict mapping result name (file stem) to ScorerMetrics.
        """
        from pyrit.common.path import SCORER_EVALS_PATH
        
        results = {}
        metrics_type = MetricsType.OBJECTIVE if isinstance(self.scorer, TrueFalseScorer) else MetricsType.HARM
        
        for dataset_config in dataset_files:
            # Collect all matching files
            csv_files = []
            for pattern in dataset_config.human_labeled_datasets_files:
                matched = list(SCORER_EVALS_PATH.glob(pattern))
                csv_files.extend(matched)
            
            if not csv_files:
                logger.warning(f"No files found for patterns {dataset_config.human_labeled_datasets_files}")
                continue
            
            # Result key is the stem of the result file
            result_key = Path(dataset_config.result_file).stem
            
            # Load each CSV as a HumanLabeledDataset and combine entries
            all_entries = []
            for csv_file in csv_files:
                dataset = HumanLabeledDataset.from_csv(
                    csv_path=csv_file,
                    metrics_type=metrics_type,
                    assistant_response_col_name=STANDARD_ASSISTANT_RESPONSE_COL,
                    human_label_col_names=[STANDARD_HUMAN_LABEL_COL],
                    objective_or_harm_col_name=STANDARD_OBJECTIVE_COL,
                    # data_type column is optional - defaults to "text" when not provided
                )
                all_entries.extend(dataset.entries)
            
            # Create combined dataset
            combined_dataset = HumanLabeledDataset(
                entries=all_entries,
                metrics_type=metrics_type,
                name=dataset_config.result_file,
                version="combined",
            )
            
            # Evaluate
            metrics = await self.run_evaluation_async(
                labeled_dataset=combined_dataset,
                num_scorer_trials=num_scorer_trials,
                add_to_registry=add_to_registry,
            )
            
            results[result_key] = metrics
        
        return results

    @abc.abstractmethod
    async def run_evaluation_async(
        self,
        labeled_dataset: HumanLabeledDataset,
        num_scorer_trials: int = 1,
        add_to_registry: bool = False,
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate the scorer against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses.
            add_to_registry (bool): Whether to add the metrics to the official evaluation results. Defaults to False. This
                should only be True when running evaluations on official datasets.

        Returns:
            ScorerMetrics: The metrics for the scorer. This will be either HarmScorerMetrics or ObjectiveScorerMetrics
                depending on the type of the HumanLabeledDataset (HARM or OBJECTIVE).
        """
        pass


class HarmScorerEvaluator(ScorerEvaluator):
    """
    A class that evaluates a harm scorer against HumanLabeledDatasets of type HARM.
    """

    async def run_evaluation_async(
        self,
        labeled_dataset: HumanLabeledDataset,
        num_scorer_trials: int = 1,
        add_to_registry: bool = False,
    ) -> HarmScorerMetrics:
        """
        Evaluate the scorer against a HumanLabeledDataset of type HARM.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            add_to_registry (bool): Whether to add the metrics to the official evaluation results. Defaults to False. This
                should only be True when running evaluations on official datasets.

        Returns:
            HarmScorerMetrics: The metrics for the harm scorer.

        Raises:
            ValueError: If the HumanLabeledDataset is not of type HARM or contains multiple harm categories.
        """
        if add_to_registry:
            logger.warning("Evaluation results functionality for harm scoring should use add_to_harm_evaluation_results(). Ignoring add_to_registry flag.")

        if labeled_dataset.metrics_type != MetricsType.HARM:
            raise ValueError("The HumanLabeledDataset must be of type HARM to evaluate a harm scorer.")

        harm_categories: Set[str] = {
            entry.harm_category for entry in labeled_dataset.entries if isinstance(entry, HarmHumanLabeledEntry)
        }
        if len(harm_categories) > 1:
            raise ValueError("Evaluating a dataset with multiple harm categories is not currently supported.")

        assistant_responses, human_scores_list, harms = [], [], []
        for index, entry in enumerate(labeled_dataset.entries):
            if not isinstance(entry, HarmHumanLabeledEntry):
                raise ValueError(
                    f"Entry at index {index} is not a HarmHumanLabeledEntry, "
                    "but the HumanLabeledDataset type is HARM."
                )
            for message in entry.conversation:
                self.scorer._memory.add_message_to_memory(request=message)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(message)
            human_scores_list.append(entry.human_scores)
            harms.append(entry.harm_category)

        # Transpose human scores list so each row is a complete set of human scores across all the responses
        # (i.e. if there are 200 responses and 3 human scores per response, the shape will be (3, 200))
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_batch_async(
                messages=assistant_responses, infer_objective_from_request=True
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)

        harm_metrics = self._compute_harm_metrics(
            all_human_scores=all_human_scores,
            all_model_scores=all_model_scores,
        )

        # Include trial scores for debugging and analysis
        harm_metrics.trial_scores = all_model_scores

        return harm_metrics

    def _compute_harm_metrics(
        self,
        *,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
    ) -> HarmScorerMetrics:
        reliability_data = np.concatenate((all_human_scores, all_model_scores))
        # Calculate the mean of human scores for each response, which is considered the gold label
        gold_scores = np.mean(all_human_scores, axis=0)
        mean_model_scores = np.mean(all_model_scores, axis=0)
        diff = mean_model_scores - gold_scores

        # Zero out tiny floating point noise
        diff[np.abs(diff) < 1e-10] = 0.0

        abs_error = np.abs(diff)
        t_statistic, p_value = cast(Tuple[float, float], ttest_1samp(diff, 0))
        metrics = {
            "mean_absolute_error": np.mean(abs_error),
            "mae_standard_error": np.std(abs_error) / np.sqrt(len(abs_error)),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "krippendorff_alpha_combined": krippendorff_alpha(
                reliability_data=reliability_data, level_of_measurement="ordinal"
            ),
        }
        if len(all_human_scores) > 1:
            metrics["krippendorff_alpha_humans"] = krippendorff_alpha(
                reliability_data=all_human_scores, level_of_measurement="ordinal"
            )
        if len(all_model_scores) > 1:
            metrics["krippendorff_alpha_model"] = krippendorff_alpha(
                reliability_data=all_model_scores, level_of_measurement="ordinal"
            )

        return HarmScorerMetrics(**metrics)


class ObjectiveScorerEvaluator(ScorerEvaluator):
    """
    A class that evaluates an objective scorer against HumanLabeledDatasets of type OBJECTIVE.
    """

    async def run_evaluation_async(
        self,
        labeled_dataset: HumanLabeledDataset,
        num_scorer_trials: int = 1,
        add_to_registry: bool = False,
    ) -> ObjectiveScorerMetrics:
        """
        Evaluate the scorer against a HumanLabeledDataset of type OBJECTIVE.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            add_to_registry (bool): Whether to add the metrics to the official evaluation results. Defaults to False. This
                should only be True when running evaluations on official datasets.

        Returns:
            ObjectiveScorerMetrics: The metrics for the objective scorer.

        Raises:
            ValueError: If the HumanLabeledDataset is not of type OBJECTIVE or contains invalid entries.
        """
        if labeled_dataset.metrics_type != MetricsType.OBJECTIVE:
            raise ValueError("The HumanLabeledDataset must be of type OBJECTIVE to evaluate an objective scorer.")
        assistant_responses, human_scores_list, objectives = [], [], []
        for index, entry in enumerate(labeled_dataset.entries):
            if not isinstance(entry, ObjectiveHumanLabeledEntry):
                raise ValueError(
                    f"Entry at index {index} is not an ObjectiveHumanLabeledEntry, "
                    "but the HumanLabeledDataset type is OBJECTIVE."
                )
            for message in entry.conversation:
                self.scorer._memory.add_message_to_memory(request=message)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(message.message_pieces[0])
            human_scores_list.append(entry.human_scores)
            objectives.append(entry.objective)

        # Transpose human scores list so each row is a complete set of human scores across all the responses
        # (i.e. if there are 200 responses and 3 human scores per response, the shape will be (3, 200))
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_batch_async(
                messages=[piece.to_message() for piece in assistant_responses],
                objectives=objectives,
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)

        objective_metrics = self._compute_objective_metrics(
            all_human_scores=all_human_scores,
            all_model_scores=all_model_scores,
        )

        # Include trial scores for debugging and analysis
        objective_metrics.trial_scores = all_model_scores

        if add_to_registry:
            try:
                from pyrit.score.scorer_evaluation.scorer_metrics_utility import add_to_objective_evaluation_results
                
                # Don't include trial_scores in evaluation results entry (too large)
                results_metrics = ObjectiveScorerMetrics(
                    accuracy=objective_metrics.accuracy,
                    accuracy_standard_error=objective_metrics.accuracy_standard_error,
                    f1_score=objective_metrics.f1_score,
                    precision=objective_metrics.precision,
                    recall=objective_metrics.recall,
                )
                
                add_to_objective_evaluation_results(
                    scorer_identifier=self.scorer.scorer_identifier,
                    metrics=results_metrics,
                    dataset_version=labeled_dataset.version,
                )
            except Exception as e:
                logger.warning(f"Failed to add metrics to evaluation results: {e}")

        return objective_metrics

    def _compute_objective_metrics(
        self,
        *,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
    ) -> ObjectiveScorerMetrics:
        # Calculate the majority vote of human scores for each response, which is considered the gold label.
        # If the vote is split, the resulting gold score will be 0 (i.e. False). Same logic is applied to model trials.
        gold_scores = np.round(np.mean(all_human_scores, axis=0))
        majority_model_scores = np.round(np.mean(all_model_scores, axis=0))

        true_positive = np.sum((gold_scores == 1) & (majority_model_scores == 1))
        false_positive = np.sum((gold_scores == 0) & (majority_model_scores == 1))
        true_negative = np.sum((gold_scores == 0) & (majority_model_scores == 0))
        false_negative = np.sum((gold_scores == 1) & (majority_model_scores == 0))
        accuracy = (true_positive + true_negative) / gold_scores.size
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "accuracy_standard_error": np.sqrt(accuracy * (1 - accuracy) / gold_scores.size),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        return ObjectiveScorerMetrics(**metrics)
