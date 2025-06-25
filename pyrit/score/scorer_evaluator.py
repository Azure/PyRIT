# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import abc
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional, Type, TypeVar, Union

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pyrit.common.path import (
    SCORER_EVALS_HARM_PATH,
    SCORER_EVALS_OBJECTIVE_PATH,
)
from pyrit.score import (
    HarmHumanLabeledEntry,
    HumanLabeledDataset,
    ObjectiveHumanLabeledEntry,
    Scorer,
)

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
    Metrics for evaluating a harm scorer against a human-labeled dataset.

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
            human and model scores. This measures the agreement between all the human raters and model scoring trials.
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
    Metrics for evaluating an objective scorer against a human-labeled dataset.

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
    A class that evaluates an LLM scorer against human-labeled datasets, calculating appropriate
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
    def from_scorer(
        cls, scorer: Scorer, metrics_type: Optional[Literal["harm", "objective"]] = None
    ) -> Union["HarmScorerEvaluator", "ObjectiveScorerEvaluator"]:
        """
        Factory method to create a ScorerEvaluator based on the type of scoring.

        Args:
            scorer (Scorer): The scorer to evaluate.
            metrics_type (Literal["harm", "objective"]): The type of scoring, either "harm" or "objective".
                If not provided, it will default to "objective" for true/false scorers and "harm" for all other
                scorers.

        Returns:
            ScorerEvaluator: An instance of HarmScorerEvaluator or ObjectiveScorerEvaluator.
        """
        if not metrics_type:
            metrics_type = "objective" if scorer.scorer_type == "true_false" else "harm"
        if metrics_type == "harm":
            return HarmScorerEvaluator(scorer=scorer)
        elif metrics_type == "objective":
            return ObjectiveScorerEvaluator(scorer=scorer)

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
        assistant_response_col: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in CSV file.

        Args:
            csv_path (str): The path to the CSV file, which will be used to construct the HumanLabeledDataset
                object.
            assistant_response_col (str): The name of the column in the CSV file that contains the assistant responses.
            human_label_col_names (List[str]): The names of the columns in the CSV file that contain the human labels.
            objective_or_harm_col_name (str): The name of the column in the CSV file that contains the objective or harm
                category associated with each response.
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
            labeled_dataset (HumanLabeledDataset): The human-labeled dataset to evaluate the scorer against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses.
            save_results (bool): Whether to save the metrics in a JSON file and the model score(s) for each response
                in a CSV file. Defaults to True.
        Returns:
            ScorerMetrics: The metrics for the scorer. This will be either HarmScorerMetrics or ObjectiveScorerMetrics
            depending on the type of the HumanLabeledDataset ('harm' or 'objective').
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
        Save the scores to a CSV file.

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
    A class that evaluates a harm scorer against human-labeled datasets of type 'harm'.
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
        assistant_response_col: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> HarmScorerMetrics:
        labeled_dataset = HumanLabeledDataset.from_csv(
            csv_path=csv_path,
            metrics_type="harm",
            assistant_responses_col_name=assistant_response_col,
            human_label_col_names=human_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
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
        Evaluate the scorer against a human-labeled dataset of type 'harm'. If save_results is True, the evaluation
        metrics and CSV file containing the model scores for each trial will be saved in the
        'dataset/score/scorer_evals/harm' directory based on the name of the HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The human-labeled dataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            save_results (bool): Whether to save the metrics and model scoring results. Defaults to True.

        Returns:
            HarmScorerMetrics: The metrics for the harm scorer.
        """
        if labeled_dataset.metrics_type != "harm":
            raise ValueError("The human-labeled dataset must be of type 'harm' to evaluate a harm scorer.")

        if len({entry.harm_category for entry in labeled_dataset.entries}) > 1:  # type: ignore
            raise ValueError("Evaluating a dataset with multiple harm categories is not currently supported.")

        assistant_responses, human_scores_list, harms = [], [], []
        for index, entry in enumerate(labeled_dataset.entries):
            if not isinstance(entry, HarmHumanLabeledEntry):
                raise ValueError(
                    f"Entry at index {index} is not a HarmHumanLabeledEntry,"
                    " but the human-labeled dataset type is 'harm'."
                )
            for request_response in entry.responses_to_score:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response.request_pieces[0])
            human_scores_list.append(entry.human_scores)
            harms.append(entry.harm_category)

        # Transpose human scores list so each row is a complete set of human scores for all the responses
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_responses_inferring_tasks_batch_async(
                request_responses=assistant_responses
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
                responses=[response.converted_value for response in assistant_responses],
                all_model_scores=all_model_scores,
                file_path=csv_results_path,
            )
            # Save the metrics to a JSON file
            with open(metrics_path, "w") as f:
                json.dump(asdict(harm_metrics), f, indent=4)

        return harm_metrics

    def _compute_harm_metrics(
        self,
        all_human_scores: np.ndarray,
        all_model_scores: np.ndarray,
    ) -> HarmScorerMetrics:
        reliability_data = np.concatenate((all_human_scores, all_model_scores))
        # Calculate the mean of human scores for each response, which is considered the gold label
        gold_scores = np.mean(all_human_scores, axis=0)
        mean_model_scores = np.mean(all_model_scores, axis=0)
        diff = mean_model_scores - gold_scores
        abs_error = np.abs(diff)
        metrics = {
            "mean_absolute_error": np.mean(abs_error),
            "mae_standard_error": np.std(abs_error) / np.sqrt(len(abs_error)),
            "t_statistic": ttest_1samp(diff, 0).statistic,
            "p_value": ttest_1samp(diff, 0).pvalue,
            "krippendorff_alpha_combined": krippendorff.alpha(
                reliability_data=reliability_data, level_of_measurement="ordinal"
            ),
        }
        if len(all_human_scores) > 1:
            metrics["krippendorff_alpha_humans"] = krippendorff.alpha(
                reliability_data=all_human_scores, level_of_measurement="ordinal"
            )
        if len(all_model_scores) > 1:
            metrics["krippendorff_alpha_model"] = krippendorff.alpha(
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
    A class that evaluates an objective scorer against human-labeled datasets of type 'objective'.
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
        assistant_response_col: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        num_scorer_trials: int = 1,
        save_results: bool = True,
        dataset_name: Optional[str] = None,
    ) -> ObjectiveScorerMetrics:
        labeled_dataset = HumanLabeledDataset.from_csv(
            csv_path=csv_path,
            metrics_type="objective",
            assistant_responses_col_name=assistant_response_col,
            human_label_col_names=human_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
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
        Evaluate the scorer against a human-labeled dataset of type 'objective'. If save_results is True, the evaluation
        metrics and CSV file containing the model scores for each trial will be saved in the
        'dataset/score/scorer_evals/objective' directory based on the name of the HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The human-labeled dataset to evaluate against.
            num_scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            save_results (bool): Whether to save the metrics and model scoring results. Defaults to True.
        Returns:
            ObjectiveScorerMetrics: The metrics for the objective scorer.
        """
        if labeled_dataset.metrics_type != "objective":
            raise ValueError("The human-labeled dataset must be of type 'objective' to evaluate an objective scorer.")
        assistant_responses, human_scores_list, objectives = [], [], []
        for index, entry in enumerate(labeled_dataset.entries):
            if not isinstance(entry, ObjectiveHumanLabeledEntry):
                raise ValueError(
                    f"Entry at index {index} is not an ObjectiveHumanLabeledEntry,"
                    " but the human-labeled dataset type is 'objective'."
                )
            for request_response in entry.responses_to_score:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response.request_pieces[0])
            human_scores_list.append(entry.human_scores)
            objectives.append(entry.objective)

        # Transpose human scores list so each row is a complete set of human scores for all the responses
        all_human_scores = np.array(human_scores_list).T

        all_model_scores_list = []
        for _ in range(num_scorer_trials):
            scores = await self.scorer.score_prompts_with_tasks_batch_async(
                request_responses=assistant_responses, tasks=objectives
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
        accuracy = (true_positive + true_negative) / len(gold_scores)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "accuracy_standard_error": np.sqrt(accuracy * (1 - accuracy) / len(gold_scores)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        return ObjectiveScorerMetrics(**metrics)

    def _get_metrics_path(self, dataset_name: str) -> Path:
        """
        Get the path to save the metrics file.

        Args:
            metrics_type (Literal["harm", "objective"]): The type of the scorer metrics, either "harm" or "objective".
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
            metrics_type (Literal["harm", "objective"]): The type of the scorer metrics, either "harm" or "objective".
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to the CSV to save the results from the LLM scoring trials.
        """
        scorer_name = type(self.scorer).__name__
        return Path(SCORER_EVALS_OBJECTIVE_PATH, f"{dataset_name}_{scorer_name}_scoring_results.csv").resolve()
