# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pyrit.common.path import SCORER_EVALS_PATH, SCORER_EVALS_HARM_PATH, SCORER_EVALS_OBJECTIVE_PATH
from pyrit.score import HarmHumanLabeledEntry, HumanLabeledDataset, ObjectiveHumanLabeledEntry, Scorer

logger = logging.getLogger(__name__)


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
    def from_json(cls, file_path: Union[str, Path]) -> "ScorerMetrics":
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
        metrics_type = data.get("type")
        if metrics_type == "harm":
            return HarmScorerMetrics(**data)
        elif metrics_type == "objective":
            return ObjectiveScorerMetrics(**data)
        else:
            raise ValueError(f"Unknown metric type: {metrics_type}")


@dataclass
class HarmScorerMetrics(ScorerMetrics):
    mean_absolute_error: float
    mae_standard_error: float
    t_statistic: float
    p_value: float
    krippendorff_alpha_combined: float
    krippendorff_alpha_humans: Optional[float] = None
    krippendorff_alpha_model: Optional[float] = None
    type: str = "harm"


@dataclass
class ObjectiveScorerMetrics(ScorerMetrics):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    type: str = "objective"


class ScorerEvaluator:
    """
    A class that evaluates an LLM scorer against a human-labeled dataset, calculating appropriate
    metrics and saving them to a file.
    """
    def __init__(self, scorer: Scorer):
        """
        Initialize the ScorerEvaluator with a scorer and policy path.

        Args:
            scorer (Scorer): The scorer to evaluate.
        """
        self.scorer = scorer

        self.scorer_name = type(scorer).__name__
        # self.metrics_dir = Path(SCORER_EVALS_PATH, "metrics", scorer_name).resolve()
        # if not os.path.exists(self.metrics_dir):
        #     self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # self.csv_results_dir = Path(SCORER_EVALS_PATH, "csv_results", scorer_name).resolve()
        # if not os.path.exists(self.csv_results_dir):
        #     self.csv_results_dir.mkdir(parents=True, exist_ok=True)

    def get_scorer_metrics(self, metrics_type: Literal["harm", "objective"], dataset_name: str) -> ScorerMetrics:
        """
        Get the metrics for the scorer if they exist.
        Args:
            metrics_type (Literal["harm", "objective"]): The type of the scorer metrics, either "harm" or "objective".
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run. This is used to
                construct the file name for the metrics.

        Returns:
            ScorerMetrics: The metrics for the scorer.
        """
        metrics_path = self._get_metrics_path(metrics_type=metrics_type, dataset_name=dataset_name)
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"{metrics_path} does not exist. Evaluation may not have been run with this harm, objective, or dataset yet."
            )
        return ScorerMetrics.from_json(metrics_path)

    async def run_evaluation_async(
        self, labeled_dataset: HumanLabeledDataset, scorer_trials: int = 1, dataset_name: Optional[str] = None, save_results: bool = True
    ) -> ScorerMetrics:
        if labeled_dataset.type == "harm":
            return await self.evaluate_harm_scorer_async(
                labeled_dataset=labeled_dataset, scorer_trials=scorer_trials, save_results=save_results
            )
        elif labeled_dataset.type == "objective":
            return await self.evaluate_objective_scorer_async(
                labeled_dataset=labeled_dataset, scorer_trials=scorer_trials, save_results=save_results
            )
        else:
            raise ValueError(f"Unsupported dataset type: {labeled_dataset.type}. Supported types are 'harm' and 'objective'.")

    async def evaluate_harm_scorer_async(
        self, 
        labeled_dataset: HumanLabeledDataset, 
        scorer_trials: int = 1,
        save_results: bool = True,
    ) -> HarmScorerMetrics:
        """
        Evaluate the scorer against a human-labeled dataset of type 'harm'. If save_results is True, the evaluation
        metrics and CSV file containing the model scores for each trial will be saved in the "scorer_evals/harm" 
        directory based on the name of the HumanLabeledDataset.

        Args:
            labeled_dataset (HumanLabeledDataset): The human-labeled dataset to evaluate against.
            scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1.
            save_results (bool): Whether to save the results to a file. Defaults to True.

        Returns:
            HarmScorerMetrics: The metrics for the harm scorer.
        """
        if labeled_dataset.type != "harm":
            raise ValueError("The human-labeled dataset must be of type 'harm' to evaluate a harm scorer.")
        if len({entry.harm_category for entry in labeled_dataset.entries}) > 1:
            logging.warning("Evaluating a dataset with multiple harm categories is not currently supported. "
                            "Scorer will use the scale passed in at instantiation for all entries.")

        assistant_responses, human_scores_list, harms = [], [], []
        for entry in labeled_dataset.entries:
            for request_response in entry.responses_to_score:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response.request_pieces[0])
            human_scores_list.append(entry.human_scores)
            if isinstance(entry, HarmHumanLabeledEntry):
                harms.append(entry.harm_category)
            else:
                raise ValueError(f"{entry} is not a HarmHumanLabeledEntry, but the human-labeled dataset type is 'harm'.")

        # Transpose human scores list so each row is a complete set of human scores for all the responses
        all_human_scores = np.array(human_scores_list).T
        # Calculate the mean of human scores for each response, which is considered the gold label
        gold_scores = np.mean(all_human_scores, axis=0)
        all_model_scores_list = []
        for trial in range(scorer_trials):
            scores = await self.scorer.score_responses_inferring_tasks_batch_async(
                request_responses=assistant_responses
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)
        reliability_data = np.concatenate((all_human_scores, all_model_scores))

        mean_model_scores = np.mean(all_model_scores, axis=0)
        diff = mean_model_scores - gold_scores
        abs_error = np.abs(diff)
        metrics = {
            "mean_absolute_error": np.mean(abs_error),
            "mae_standard_error": np.std(abs_error) / np.sqrt(len(abs_error)),
            # t-statistic and p-value via 1-sample t-test for the null hypothesis that the mean difference is 0
            # High positive t-statistic and low p-value indicate model scores are typically higher than human scores
            "t_statistic": ttest_1samp(diff, 0).statistic,
            "p_value": ttest_1samp(diff, 0).pvalue,
            "krippendorff_alpha_combined": krippendorff.alpha(
                reliability_data=reliability_data, level_of_measurement="ordinal"
            ),
            "type": "harm",
        }
        if len(all_human_scores) > 1:
            metrics["krippendorff_alpha_humans"] = krippendorff.alpha(
                reliability_data=all_human_scores, level_of_measurement="ordinal"
            )
        if len(all_model_scores) > 1:
            metrics["krippendorff_alpha_model"] = krippendorff.alpha(
                reliability_data=all_model_scores, level_of_measurement="ordinal"
            )
        harm_metrics = HarmScorerMetrics(**metrics)
        if save_results:
            metrics_path = self._get_metrics_path(metrics_type="harm", dataset_name=labeled_dataset.name)
            csv_results_path = self._get_csv_results_path(metrics_type="harm", dataset_name=labeled_dataset.name)
            self._save_model_scores_to_csv(
                responses=[response.converted_value for response in assistant_responses],
                all_model_scores=all_model_scores,
                file_path=csv_results_path,
            )
            # Save the metrics to a JSON file
            with open(metrics_path, "w") as f:
                json.dump(asdict(harm_metrics), f, indent=4)

        return harm_metrics

    async def evaluate_objective_scorer_async(
        self, labeled_dataset: HumanLabeledDataset, scorer_trials: int = 1, dataset_name: Optional[str] = None, save_results: bool = True
    ) -> ObjectiveScorerMetrics:
        pass

    async def run_evaluation_from_csv_async(
        self,
        csv_path: Union[str, Path],
        dataset_name: str,
        type: Literal["harm", "objective"],
        assistant_response_col: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        scorer_trials: int = 1,
        save_results: bool = True,
    ) -> ScorerMetrics:
        """
        Run the evaluation for the scorer/policy combination on the passed in CSV file.

        Args:
            csv_path (str): The path to the CSV file.
            scorer_trials (int): The number of trials to run the scorer on all responses.
            save_results (bool): Whether to save the results to a file.

        Returns:
            ScorerMetrics: The metrics for the scorer.
        """
        labeled_dataset = HumanLabeledDataset.from_csv(
            csv_path=csv_path,
            dataset_name=dataset_name,
            type=type,
            assistant_responses_col_name=assistant_response_col,
            human_label_col_names=human_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
        )
        metrics = await self.run_evaluation_async(
            labeled_dataset=labeled_dataset, scorer_trials=scorer_trials, save_results=save_results
        )

        return metrics

    def _save_model_scores_to_csv(
        self,
        responses: List[str],
        all_model_scores: np.ndarray,
        file_path: Path,
    ):
        """
        Save the scores to a CSV file.

        Args:
            responses (List[str]): The assistant responses.
            all_model_scores (np.ndarray): The scores for each trial.
            file_path (Path): The path save the model scoring trials to.
        """
        cols_dict = {"assistant_responses": responses}
        for trial, scores in enumerate(all_model_scores):
            cols_dict[f"trial {trial+1}"] = scores

        scores_df = pd.DataFrame(cols_dict)
        scores_df.to_csv(file_path, index=False)

    def _get_metrics_path(self, metrics_type: Literal["harm", "objective"], dataset_name: str) -> Path:
        """
        Get the path to save the metrics file.

        Args:
            metrics_type (Literal["harm", "objective"]): The type of the scorer metrics, either "harm" or "objective".
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to save the metrics file.
        """
        return Path(SCORER_EVALS_PATH, metrics_type, f"{dataset_name}_{self.scorer_name}_metrics.json").resolve()

    def _get_csv_results_path(self, metrics_type: Literal["harm", "objective"], dataset_name: str) -> Path:
        """
        Get the path to save the CSV results file.

        Args:
            metrics_type (Literal["harm", "objective"]): The type of the scorer metrics, either "harm" or "objective".
            dataset_name (str): The name of the HumanLabeledDataset on which evaluation was run.

        Returns:
            Path: The path to the CSV to save the results from the LLM scoring trials.
        """
        return Path(SCORER_EVALS_PATH, metrics_type, f"{dataset_name}_{self.scorer_name}_scoring_results.csv").resolve()

