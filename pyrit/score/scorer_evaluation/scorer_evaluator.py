# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pyrit.common.path import (
    SCORER_EVALS_HARM_PATH,
    SCORER_EVALS_OBJECTIVE_PATH,
)
from pyrit.models.prompt_request_response import PromptRequestResponse
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


@dataclass
class ScorerMetrics:
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
            json_path (str): The path to the JSON file.

        Returns:
            ScorerMetrics: An instance of ScorerMetrics with the loaded data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
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
    """

    accuracy: float
    accuracy_standard_error: float
    f1_score: float
    precision: float
    recall: float


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
        Factory method to create a ScorerEvaluator based on the type of scoring.

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

    @abc.abstractmethod
    def get_scorer_metrics(self, dataset_name: str) -> ScorerMetrics:
        """
        Get the metrics for the scorer in the 'dataset/score/scorer_evals' directory based on the dataset name.

        Args:
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            ScorerMetrics: The metrics for the scorer.
        """
        pass

    @abc.abstractmethod
    async def run_evaluation_from_csv_async(
        self,
        csv_path: Union[str, Path],
        assistant_response_col_name: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        assistant_response_data_type_col_name: Optional[str] = None,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in CSV file.

        Args:
            csv_path (str): The path to the CSV file, which will be used to construct the HumanLabeledDataset
                object.
            assistant_response_col_name (str): The name of the column in the CSV file that contains the assistant
                responses.
            human_label_col_names (List[str]): The names of the columns in the CSV file that contain the human labels.
            objective_or_harm_col_name (str): The name of the column in the CSV file that contains the objective or harm
                category associated with each response.
            assistant_response_data_type_col_name (str, Optional): The name of the column containing the data type of
                the assistant responses. If not specified, it is assumed that the responses are text.
            num_scorer_trials (int): The number of trials to run the scorer on all responses.
            save_results (bool): Whether to save the metrics in a JSON file and the model score(s) for each response
                in a CSV file. Defaults to True.
            dataset_name (str, Optional): The name of the dataset. If not provided, it will be inferred from the CSV
                file name. This is used to inform the name of the metrics file and model scoring results CSV to save
                in the 'scorer_evals' directory.

        Returns:
            ScorerMetrics: The metrics for the scorer.
        """
        pass

    @abc.abstractmethod
    async def run_evaluation_async(
        self, labeled_dataset: HumanLabeledDataset, num_scorer_trials: int = 1, save_results: bool = True
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate the scorer against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses.
            save_results (bool): Whether to save the metrics in a JSON file and the model score(s) for each response
                in a CSV file. Defaults to True.

        Returns:
            ScorerMetrics: The metrics for the scorer. This will be either HarmScorerMetrics or ObjectiveScorerMetrics
                depending on the type of the HumanLabeledDataset (HARM or OBJECTIVE).
        """
        pass

    def _save_model_scores_to_csv(
        self,
        objectives_or_harms: List[str],
        responses: List[str],
        all_model_scores: np.ndarray,
        file_path: Path,
    ):
        """
        Save the scores generated by the LLM scorer during evaluation to a CSV file.

        Args:
            objectives_or_harms (List[str]): The objectives or harms associated with each response.
            responses (List[str]): The assistant responses.
            all_model_scores (np.ndarray): The scores for each trial.
            file_path (Path): The path save the model scoring trials to.
        """
        cols_dict = {"objective_or_harm": objectives_or_harms, "assistant_response": responses}
        for trial, scores in enumerate(all_model_scores):
            cols_dict[f"trial {trial+1}"] = scores

        scores_df = pd.DataFrame(cols_dict)
        scores_df.to_csv(file_path, index=False)


class HarmScorerEvaluator(ScorerEvaluator):
    """
    A class that evaluates a harm scorer against HumanLabeledDatasets of type HARM.
    """

    def get_scorer_metrics(self, dataset_name) -> HarmScorerMetrics:
        metrics_path = self._get_metrics_path(dataset_name=dataset_name)
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"{metrics_path} does not exist. Evaluation may not have been run with this dataset yet."
            )
        return HarmScorerMetrics.from_json(metrics_path)

    async def run_evaluation_from_csv_async(
        self,
        csv_path: Union[str, Path],
        assistant_response_col_name: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        assistant_response_data_type_col_name: Optional[str] = None,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> HarmScorerMetrics:
        labeled_dataset = HumanLabeledDataset.from_csv(
            csv_path=csv_path,
            metrics_type=MetricsType.HARM,
            assistant_response_col_name=assistant_response_col_name,
            human_label_col_names=human_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
            assistant_response_data_type_col_name=assistant_response_data_type_col_name,
            dataset_name=dataset_name,
        )
        metrics = await self.run_evaluation_async(
            labeled_dataset=labeled_dataset, num_scorer_trials=num_scorer_trials, save_results=save_results
        )

        return metrics

    async def run_evaluation_async(
        self,
        labeled_dataset: HumanLabeledDataset,
        num_scorer_trials: int = 1,
        save_results: bool = True,
    ) -> HarmScorerMetrics:
        """
        Evaluate the scorer against a HumanLabeledDataset of type HARM. If save_results is True, the evaluation
        metrics and CSV file containing the LLM-produced scores across all trials will be saved in the
        'dataset/score/scorer_evals/harm' directory based on the name of the HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            save_results (bool): Whether to save the metrics and model scoring results. Defaults to True.

        Returns:
            HarmScorerMetrics: The metrics for the harm scorer.
        """
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
                    f"Entry at index {index} is not a HarmHumanLabeledEntry,"
                    " but the HumanLabeledDataset type is HARM."
                )
            for request_response in entry.conversation:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response)
            human_scores_list.append(entry.human_scores)
            harms.append(entry.harm_category)

        # Transpose human scores list so each row is a complete set of human scores across all the responses
        # (i.e. if there are 200 responses and 3 human scores per response, the shape will be (3, 200))
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_batch_async(
                request_responses=assistant_responses, infer_objective_from_request=True
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)

        harm_metrics = self._compute_harm_metrics(
            all_human_scores=all_human_scores,
            all_model_scores=all_model_scores,
        )

        if save_results:
            metrics_path = self._get_metrics_path(dataset_name=labeled_dataset.name)
            csv_results_path = self._get_csv_results_path(dataset_name=labeled_dataset.name)
            self._save_model_scores_to_csv(
                objectives_or_harms=harms,
                responses=PromptRequestResponse.get_all_values(assistant_responses),
                all_model_scores=all_model_scores,
                file_path=csv_results_path,
            )
            # Save the metrics to a JSON file
            with open(metrics_path, "w") as f:
                json.dump(asdict(harm_metrics), f, indent=4)

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

    def _get_metrics_path(self, dataset_name: str) -> Path:
        """
        Get the path to save the metrics file.

        Args:
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to save the metrics file.
        """
        scorer_name = type(self.scorer).__name__
        return Path(SCORER_EVALS_HARM_PATH, f"{dataset_name}_{scorer_name}_metrics.json").resolve()

    def _get_csv_results_path(self, dataset_name: str) -> Path:
        """
        Get the path to save the CSV results file.

        Args:
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to the CSV to save the results from the LLM scoring trials.
        """
        scorer_name = type(self.scorer).__name__
        return Path(SCORER_EVALS_HARM_PATH, f"{dataset_name}_{scorer_name}_scoring_results.csv").resolve()


class ObjectiveScorerEvaluator(ScorerEvaluator):
    """
    A class that evaluates an objective scorer against HumanLabeledDatasets of type OBJECTIVE.
    """

    def get_scorer_metrics(self, dataset_name: str) -> ObjectiveScorerMetrics:
        metrics_path = self._get_metrics_path(dataset_name=dataset_name)
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"{metrics_path} does not exist. Evaluation may not have been run with this dataset yet."
            )
        return ObjectiveScorerMetrics.from_json(metrics_path)

    async def run_evaluation_from_csv_async(
        self,
        csv_path: Union[str, Path],
        assistant_response_col_name: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        assistant_response_data_type_col_name: Optional[str] = None,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> ObjectiveScorerMetrics:
        labeled_dataset = HumanLabeledDataset.from_csv(
            csv_path=csv_path,
            metrics_type=MetricsType.OBJECTIVE,
            assistant_response_col_name=assistant_response_col_name,
            human_label_col_names=human_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
            assistant_response_data_type_col_name=assistant_response_data_type_col_name,
            dataset_name=dataset_name,
        )
        metrics = await self.run_evaluation_async(
            labeled_dataset=labeled_dataset, num_scorer_trials=num_scorer_trials, save_results=save_results
        )

        return metrics

    async def run_evaluation_async(
        self, labeled_dataset: HumanLabeledDataset, num_scorer_trials: int = 1, save_results: bool = True
    ) -> ObjectiveScorerMetrics:
        """
        Evaluate the scorer against a HumanLabeledDataset of type OBJECTIVE. If save_results is True, the evaluation
        metrics and CSV file containing the LLM-produced scores across all trials will be saved in the
        'dataset/score/scorer_evals/objective' directory based on the name of the HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The HumanLabeledDataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            save_results (bool): Whether to save the metrics and model scoring results. Defaults to True.
        Returns:
            ObjectiveScorerMetrics: The metrics for the objective scorer.
        """
        if labeled_dataset.metrics_type != MetricsType.OBJECTIVE:
            raise ValueError("The HumanLabeledDataset must be of type OBJECTIVE to evaluate an objective scorer.")
        assistant_responses, human_scores_list, objectives = [], [], []
        for index, entry in enumerate(labeled_dataset.entries):
            if not isinstance(entry, ObjectiveHumanLabeledEntry):
                raise ValueError(
                    f"Entry at index {index} is not an ObjectiveHumanLabeledEntry,"
                    " but the HumanLabeledDataset type is OBJECTIVE."
                )
            for request_response in entry.conversation:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response.request_pieces[0])
            human_scores_list.append(entry.human_scores)
            objectives.append(entry.objective)

        # Transpose human scores list so each row is a complete set of human scores across all the responses
        # (i.e. if there are 200 responses and 3 human scores per response, the shape will be (3, 200))
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_batch_async(
                request_responses=[piece.to_prompt_request_response() for piece in assistant_responses],
                objectives=objectives,
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)

        objective_metrics = self._compute_objective_metrics(
            all_human_scores=all_human_scores,
            all_model_scores=all_model_scores,
        )
        if save_results:
            metrics_path = self._get_metrics_path(dataset_name=labeled_dataset.name)
            csv_results_path = self._get_csv_results_path(dataset_name=labeled_dataset.name)
            self._save_model_scores_to_csv(
                objectives_or_harms=objectives,
                responses=[response.converted_value for response in assistant_responses],
                all_model_scores=all_model_scores,
                file_path=csv_results_path,
            )
            # Save the metrics to a JSON file
            with open(metrics_path, "w") as f:
                json.dump(asdict(objective_metrics), f, indent=4)

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

    def _get_metrics_path(self, dataset_name: str) -> Path:
        """
        Get the path to save the metrics file.

        Args:
            metrics_type (MetricsType): The type of the scorer metrics, either HARM or OBJECTIVE.
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to save the metrics file.
        """
        scorer_name = type(self.scorer).__name__
        return Path(SCORER_EVALS_OBJECTIVE_PATH, f"{dataset_name}_{scorer_name}_metrics.json").resolve()

    def _get_csv_results_path(self, dataset_name: str) -> Path:
        """
        Get the path to save the CSV results file.

        Args:
            metrics_type (MetricsType): The type of the scorer metrics, either HARM or OBJECTIVE.
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to the CSV to save the results from the LLM scoring trials.
        """
        scorer_name = type(self.scorer).__name__
        return Path(SCORER_EVALS_OBJECTIVE_PATH, f"{dataset_name}_{scorer_name}_scoring_results.csv").resolve()
