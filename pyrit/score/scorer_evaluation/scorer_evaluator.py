# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
from scipy.stats import ttest_1samp

from pyrit.score import Scorer
from pyrit.score.scorer_evaluation.human_labeled_dataset import (
    HarmHumanLabeledEntry,
    HumanLabeledDataset,
    ObjectiveHumanLabeledEntry,
)
from pyrit.score.scorer_evaluation.metrics_type import MetricsType
from pyrit.score.scorer_evaluation.scorer_metrics import (
    HarmScorerMetrics,
    ObjectiveScorerMetrics,
    ScorerMetrics,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.common.path import SCORER_EVALS_PATH

from pyrit.score.scorer_evaluation.krippendorff import krippendorff_alpha
from pyrit.score.scorer_evaluation.scorer_metrics_io import add_evaluation_results


logger = logging.getLogger(__name__)

# Standard column names for evaluation datasets
STANDARD_HUMAN_LABEL_COL = "human_score"
STANDARD_OBJECTIVE_COL = "objective"
STANDARD_HARM_COL = "harm_category"
STANDARD_ASSISTANT_RESPONSE_COL = "assistant_response"
STANDARD_DATA_TYPE_COL = "data_type"


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


class ScorerEvaluator(abc.ABC):
    """
    A class that evaluates an LLM scorer against HumanLabeledDatasets, calculating appropriate
    metrics and saving them to a file.
    """

    # Subclasses must define the expected metrics type
    expected_metrics_type: MetricsType

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

    async def run_evaluation_async(
        self,
        *,
        dataset_files: List[ScorerEvalDatasetFiles],
        num_scorer_trials: int = 3,
        add_to_registry: bool = False,
        max_concurrency: int = 10,
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
            max_concurrency: Maximum number of concurrent scoring requests. Defaults to 10.
        
        Returns:
            Dict mapping result name (file stem) to ScorerMetrics.
        """
        
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
            dataset_versions = set()
            for csv_file in csv_files:
                dataset = HumanLabeledDataset.from_csv(
                    csv_path=csv_file,
                    metrics_type=metrics_type,
                )
                all_entries.extend(dataset.entries)
                dataset_versions.add(dataset.version)
            
            # Concatenate unique versions, sorted for consistency
            combined_version = "_".join(sorted(dataset_versions))
            
            # Create combined dataset
            combined_dataset = HumanLabeledDataset(
                entries=all_entries,
                metrics_type=metrics_type,
                name=dataset_config.result_file,
                version=combined_version,
            )
            
            # Evaluate
            metrics = await self._run_evaluation_async(
                labeled_dataset=combined_dataset,
                num_scorer_trials=num_scorer_trials,
                max_concurrency=max_concurrency,
            )
            
            # Handle registry writing if requested
            if add_to_registry:
                self._write_metrics_to_registry(
                    metrics=metrics,
                    labeled_dataset=combined_dataset,
                    result_file_path=SCORER_EVALS_PATH / dataset_config.result_file,
                )
            
            results[result_key] = metrics
        
        return results

    async def _run_evaluation_async(
        self,
        labeled_dataset: HumanLabeledDataset,
        num_scorer_trials: int = 1,
        max_concurrency: int = 10,
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in HumanLabeledDataset.
        
        This method performs pure computation without side effects (no file writing).
        Use run_evaluation_async with add_to_registry=True to write results to files.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate the scorer against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses.
            max_concurrency (int): Maximum number of concurrent scoring requests. Defaults to 10.

        Returns:
            ScorerMetrics: The metrics for the scorer. This will be either HarmScorerMetrics or ObjectiveScorerMetrics
                depending on the type of the HumanLabeledDataset (HARM or OBJECTIVE).
        """
        # Validate dataset and extract data
        assistant_responses, human_scores_list, objectives = self._validate_and_extract_data(labeled_dataset)

        # Transpose human scores so each row is a complete set of scores across all responses
        all_human_scores = np.array(human_scores_list).T

        # Run scoring trials
        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_batch_async(
                messages=assistant_responses,
                objectives=objectives,
                batch_size=max_concurrency,
                infer_objective_from_request=True,
            )
            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)

        # Compute metrics using subclass implementation
        metrics = self._compute_metrics(
            all_human_scores=all_human_scores,
            all_model_scores=all_model_scores,
            num_scorer_trials=num_scorer_trials,
        )

        # Include trial scores for debugging and analysis
        metrics.trial_scores = all_model_scores

        return metrics

    @abc.abstractmethod
    def _validate_and_extract_data(
        self,
        labeled_dataset: HumanLabeledDataset,
    ) -> Tuple[List, List[List[float]], Optional[List[str]]]:
        """
        Validate the dataset and extract data for evaluation.

        Args:
            labeled_dataset: The dataset to validate and extract from.

        Returns:
            Tuple of (assistant_responses, human_scores_list, objectives).
            objectives may be None for harm scoring.

        Raises:
            ValueError: If the dataset is invalid for this evaluator.
        """
        pass

    @abc.abstractmethod
    def _compute_metrics(
        self,
        *,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
        num_scorer_trials: int,
    ) -> ScorerMetrics:
        """
        Compute evaluation metrics from human and model scores.

        Args:
            all_human_scores: Array of human scores, shape (num_raters, num_responses).
            all_model_scores: Array of model scores, shape (num_trials, num_responses).
            num_scorer_trials: Number of scoring trials that were performed.

        Returns:
            ScorerMetrics subclass with computed metrics.
        """
        pass

    def _write_metrics_to_registry(
        self,
        *,
        metrics: ScorerMetrics,
        labeled_dataset: HumanLabeledDataset,
        result_file_path: Path,
    ) -> None:
        """
        Write metrics to the evaluation registry file.
        
        Creates a version of metrics without trial_scores (too large for registry)
        and writes to the specified file path.

        Args:
            metrics (ScorerMetrics): The computed metrics.
            labeled_dataset (HumanLabeledDataset): The dataset that was evaluated.
            result_file_path (Path): The full path to the result file.
        """

        try:
            # Extract harm_category if this is a HarmScorerMetrics
            harm_category = None
            if isinstance(metrics, HarmScorerMetrics):
                for entry in labeled_dataset.entries:
                    if isinstance(entry, HarmHumanLabeledEntry):
                        harm_category = entry.harm_category
                        break
                
                if harm_category is None:
                    raise ValueError("Could not extract harm_category from HarmScorerMetrics dataset")
            
            # Create metrics without trial_scores (too large for registry)
            metrics_dict = asdict(metrics)
            metrics_dict.pop("trial_scores", None)
            registry_metrics = type(metrics)(**metrics_dict)
            
            add_evaluation_results(
                file_path=result_file_path,
                scorer_identifier=self.scorer.scorer_identifier,
                metrics=registry_metrics,
                dataset_version=labeled_dataset.version,
                harm_category=harm_category,
            )
        except Exception as e:
            logger.warning(f"Failed to add metrics to evaluation results: {e}")


class HarmScorerEvaluator(ScorerEvaluator):
    """
    A class that evaluates a harm scorer against HumanLabeledDatasets of type HARM.
    """

    expected_metrics_type = MetricsType.HARM

    def _validate_and_extract_data(
        self,
        labeled_dataset: HumanLabeledDataset,
    ) -> Tuple[List, List[List[float]], Optional[List[str]]]:
        """
        Validate harm dataset and extract evaluation data.

        Args:
            labeled_dataset: The dataset to validate and extract from.

        Returns:
            Tuple of (assistant_responses, human_scores_list, None).
            objectives is None for harm scoring (uses infer_objective_from_request).

        Raises:
            ValueError: If dataset is not HARM type or has multiple harm categories.
        """
        if labeled_dataset.metrics_type != MetricsType.HARM:
            raise ValueError("The HumanLabeledDataset must be of type HARM to evaluate a harm scorer.")

        labeled_dataset.validate()

        assistant_responses: List = []
        human_scores_list: List[List[float]] = []

        for entry in labeled_dataset.entries:
            harm_entry = cast(HarmHumanLabeledEntry, entry)
            for message in harm_entry.conversation:
                self.scorer._memory.add_message_to_memory(request=message)
                assistant_responses.append(message)
            human_scores_list.append(harm_entry.human_scores)

        return assistant_responses, human_scores_list, None

    def _compute_metrics(
        self,
        *,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
        num_scorer_trials: int,
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
            "num_scorer_trials": num_scorer_trials,
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

    expected_metrics_type = MetricsType.OBJECTIVE

    def _validate_and_extract_data(
        self,
        labeled_dataset: HumanLabeledDataset,
    ) -> Tuple[List, List[List[float]], Optional[List[str]]]:
        """
        Validate objective dataset and extract evaluation data.

        Args:
            labeled_dataset: The dataset to validate and extract from.

        Returns:
            Tuple of (assistant_responses, human_scores_list, objectives).

        Raises:
            ValueError: If dataset is not OBJECTIVE type or contains invalid entries.
        """
        if labeled_dataset.metrics_type != MetricsType.OBJECTIVE:
            raise ValueError("The HumanLabeledDataset must be of type OBJECTIVE to evaluate an objective scorer.")

        labeled_dataset.validate()

        assistant_responses: List = []
        human_scores_list: List[List[float]] = []
        objectives: List[str] = []

        for entry in labeled_dataset.entries:
            objective_entry = cast(ObjectiveHumanLabeledEntry, entry)
            for message in objective_entry.conversation:
                self.scorer._memory.add_message_to_memory(request=message)
                assistant_responses.append(message)
            human_scores_list.append([float(score) for score in objective_entry.human_scores])
            objectives.append(objective_entry.objective)

        return assistant_responses, human_scores_list, objectives

    def _compute_metrics(
        self,
        *,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
        num_scorer_trials: int,
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
            "num_scorer_trials": num_scorer_trials,
        }

        return ObjectiveScorerMetrics(**metrics)
