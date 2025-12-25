# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import numpy as np

from pyrit.common.utils import verify_and_resolve_path

T = TypeVar("T", bound="ScorerMetrics")


@dataclass
class ScorerMetrics:
    """
    Base dataclass for storing scorer evaluation metrics.

    This class provides methods for serializing metrics to JSON and loading them from JSON files.
    
    Args:
        num_responses (int): Total number of responses evaluated.
        num_human_raters (int): Number of human raters who scored the responses.
        num_scorer_trials (int): Number of times the model scorer was run. Defaults to 1.
        dataset_name (str, optional): Name of the dataset used for evaluation.
        dataset_version (str, optional): Version of the dataset for reproducibility.
        trial_scores (np.ndarray, optional): Raw scores from each trial for debugging.
    """
    num_responses: int
    num_human_raters: int
    num_scorer_trials: int = field(default=1, kw_only=True)
    dataset_name: Optional[str] = field(default=None, kw_only=True)
    dataset_version: Optional[str] = field(default=None, kw_only=True)
    trial_scores: Optional[np.ndarray] = field(default=None, kw_only=True)

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
        
        # Extract metrics from nested structure (always under "metrics" key in evaluation result files)
        metrics_data = data.get("metrics", data)
        
        return cls(**metrics_data)


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
        harm_category (str, optional): The harm category being evaluated (e.g., "hate_speech", "violence").
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
    harm_category: Optional[str] = field(default=None, kw_only=True)
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
        trial_scores (Optional[np.ndarray]): The raw scores from each trial. Shape is (num_trials, num_responses).
            Useful for debugging and analyzing scorer variance.
    """

    accuracy: float
    accuracy_standard_error: float
    f1_score: float
    precision: float
    recall: float
