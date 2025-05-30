# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional, Type, TypeVar, Union

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pyrit.common.path import SCORER_EVALS_PATH
from pyrit.score import GoldDataset, Scorer
from pyrit.score.gold_dataset import HarmGoldDatasetEntry, ObjectiveGoldDatasetEntry


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
        metric_type = data.get("type")
        if metric_type == "harm":
            return HarmScorerMetrics(**data)
        elif metric_type == "objective":
            return ObjectiveScorerMetrics(**data)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")


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
    A class that evaluates an LLM scorer against a gold dataset, calculating appropriate
    metrics and saving them to a file.
    """

    def __init__(self, scorer: Scorer, custom_file_name: Optional[str] = None):
        """
        Initialize the ScorerEvaluator with a scorer and policy path.

        Args:
            scorer (Scorer): The scorer to evaluate.
            custom_file_name (Optional[str]): A custom file name to save the metrics under. If not provided, the
                metrics will be saved under the GoldDataset's harm category or objective name. This parameter is
                necessary if the GoldDataset does not have a harm category or objective name (e.g. it has multiple
                harm categories or objectives).
        """
        self.scorer = scorer
        self.custom_file_name = custom_file_name

        scorer_name = type(scorer).__name__
        self.metrics_dir = Path(SCORER_EVALS_PATH, "metrics", scorer_name).resolve()
        if not os.path.exists(self.metrics_dir):
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_results_dir = Path(SCORER_EVALS_PATH, "csv_results", scorer_name).resolve()
        if not os.path.exists(self.csv_results_dir):
            self.csv_results_dir.mkdir(parents=True, exist_ok=True)

        if custom_file_name:
            self.metrics_path = Path(self.metrics_dir, f"{custom_file_name}.json").resolve()
            # self.metrics_path = Path(SCORER_EVALS_PATH, "metrics", scorer_name, policy_name).resolve()
            self.csv_results_path = Path(self.csv_results_dir, f"{custom_file_name}.csv").resolve()

    def get_scorer_metrics(self, file_name: Optional[str] = None) -> ScorerMetrics:
        """
        Get the metrics for the scorer if they exist.
        Args:
            file_name (Optional[str]): The name of the file to load the metrics from. This will often be a harm category
                or objective. If not provided, it will use what is saved in the `self.metrics_path` attribute.

        Returns:
            ScorerMetrics: The metrics for the scorer.
        """
        if file_name:
            metrics_path = Path(SCORER_EVALS_PATH, "metrics", type(self.scorer).__name__, f"{file_name}.json").resolve()
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(
                    f"{metrics_path} does not exist. Evaluation may not have been run with this harm or objective yet."
                )
            return ScorerMetrics.from_json(metrics_path)
        if not self.metrics_path:
            raise ValueError(f"Please run evaluation first or pass in harm_or_objective.")

        return ScorerMetrics.from_json(self.metrics_path)

    async def run_evaluation_async(
        self, gold_dataset: GoldDataset, scorer_trials: int = 1, save_results: bool = True
    ) -> ScorerMetrics:
        if gold_dataset.type == "harm":
            return await self.evaluate_harm_scorer_async(
                gold_dataset=gold_dataset, scorer_trials=scorer_trials, save_results=save_results
            )
        elif gold_dataset.type == "objective":
            return await self.evaluate_objective_scorer_async(
                gold_dataset=gold_dataset, scorer_trials=scorer_trials, save_results=save_results
            )

    async def evaluate_harm_scorer_async(
        self, gold_dataset: GoldDataset, scorer_trials: int = 1, save_results: bool = True
    ) -> HarmScorerMetrics:
        """
        Evaluate the scorer against a gold dataset of type "harm". If save_results is True, the evaluation metrics
        will be saved in the the `self.metrics_dir` directory, and the model scores will be saved in the
        `self.csv_results_dir` directory. The file names will be based on the custom_file_name if provided
        upon instantiation of the ScorerEvaluator, or the top-level harm category of the
        GoldDataset if not provided.

        Args:
            gold_dataset (GoldDataset): The gold dataset to evaluate against.
            scorer_trials (int): The number of trials to run the scorer on all responses. Defaults to 1

        Returns:
            HarmScorerMetrics: The metrics for the harm scorer.
        """
        if gold_dataset.type != "harm":
            raise ValueError("The gold dataset must be of type 'harm' to evaluate a harm scorer.")

        if save_results:
            file_name = self.custom_file_name or gold_dataset.harm_category or gold_dataset.objective
            if not file_name:
                raise ValueError(
                    "Results cannot be saved without a custom file name when the GoldDataset "
                    "does not have a top-level harm category."
                )
            self.metrics_path = Path(self.metrics_dir, f"{file_name}.json").resolve()
            self.csv_results_path = Path(self.csv_results_dir, f"{file_name}.csv").resolve()

        assistant_responses, gold_scores_list, harms = [], [], []
        for entry in gold_dataset.entries:
            for request_response in entry.responses_to_score:
                self.scorer._memory.add_request_response_to_memory(request=request_response)
                # Logic may need to change for multi-turn scoring
                assistant_responses.append(request_response.request_pieces[0])
            gold_scores_list.append(entry.human_scores)
            if isinstance(entry, HarmGoldDatasetEntry):
                harms.append(entry.harm_category)
            else:
                raise ValueError(f"{entry} is not a HarmGoldDatasetEntry, but the gold dataset type is 'harm'.")

        # Transpose gold scores list so each row is a complete set of human scores for all the responses
        all_gold_scores = np.array(gold_scores_list).T
        mean_gold_scores = np.mean(all_gold_scores, axis=0)
        all_model_scores_list = []
        for trial in range(scorer_trials):
            scores = await self.scorer.score_prompts_with_harm_categories_batch_async(
                request_responses=assistant_responses, harm_categories=harms
            )

            score_values = [score.get_value() for score in scores]
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)
        reliability_data = np.concatenate((all_gold_scores, all_model_scores))

        average_model_scores = np.mean(all_model_scores, axis=0)
        diff = average_model_scores - mean_gold_scores
        abs_error = np.abs(diff)
        metrics = {
            "mean_absolute_error": np.mean(abs_error),
            "mae_standard_error": np.std(abs_error) / np.sqrt(len(abs_error)),
            # t-statistic and p-value via 1-sample t-test for the null hypothesis that the mean difference is 0
            # High positive t-statistic and low p-value indicate model scores are typically higher than human scores
            "t_statistic": ttest_1samp(diff, 0).statistic,
            "p_value": ttest_1samp(diff, 0).pvalue,
            "krippendorff_alpha_combined": krippendorff.alpha(
                reliability_data=reliability_data, value_domain=np.linspace(0, 1, 5), level_of_measurement="ordinal"
            ),
            "type": "harm",
        }
        if len(all_gold_scores) > 1:
            metrics["krippendorff_alpha_humans"] = krippendorff.alpha(
                reliability_data=all_gold_scores, value_domain=np.linspace(0, 1, 5), level_of_measurement="ordinal"
            )
        if len(all_model_scores) > 1:
            metrics["krippendorff_alpha_model"] = krippendorff.alpha(
                reliability_data=all_model_scores, value_domain=np.linspace(0, 1, 5), level_of_measurement="ordinal"
            )
        if save_results:
            self._save_model_scores_to_csv(
                responses=[response.converted_value for response in assistant_responses],
                all_model_scores=all_model_scores,
                file_path=self.csv_results_path,
            )
            # Save the metrics to a JSON file
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

        return HarmScorerMetrics(**metrics)

    async def evaluate_objective_scorer_async(
        self, gold_dataset: GoldDataset, scorer_trials: int = 1, save_results: bool = True
    ) -> ObjectiveScorerMetrics:
        pass

    async def run_evaluation_from_csv_async(
        self,
        csv_path: Union[str, Path],
        type: Literal["harm", "objective"],
        assistant_response_col: str,
        gold_label_col_names: List[str],
        scorer_trials: int = 1,
        objective_or_harm_col_name: Optional[str] = None,
        top_level_harm: Optional[str] = None,
        top_level_objective: Optional[str] = None,
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
        gold_dataset = GoldDataset.from_csv(
            csv_path=csv_path,
            type=type,
            assistant_response_col=assistant_response_col,
            gold_label_col_names=gold_label_col_names,
            objective_or_harm_col_name=objective_or_harm_col_name,
            top_level_harm=top_level_harm,
            top_level_objective=top_level_objective,
        )
        metrics = await self.run_evaluation_async(
            gold_dataset=gold_dataset, scorer_trials=scorer_trials, save_results=save_results
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
            file_name (Optional[str]): The name of the file to save the scores to. If not provided, it will use
                the `self.csv_results_path` attribute.
        """
        cols_dict = {"assistant_responses": responses}
        for trial, scores in enumerate(all_model_scores):
            cols_dict[f"trial {trial+1}"] = scores

        scores_df = pd.DataFrame(cols_dict)
        scores_df.to_csv(file_path, index=False)
