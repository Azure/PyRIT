# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, cast, get_args

import pandas as pd

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.models.literals import PromptDataType
from pyrit.score import MetricsType

logger = logging.getLogger(__name__)


@dataclass
class HumanLabeledEntry:
    """
    A class that represents an entry in a dataset of assistant responses that have been scored by humans. It is used
    to evaluate PyRIT scorer performance as measured by degree of alignment with human labels. This class includes
    the PromptRequestResponses and a list of human-assigned scores, which are floats between 0.0 and 1.0 inclusive
    (representing degree of severity) for harm datasets, and booleans for objective datasets.

    Parameters:
        conversation (List[PromptRequestResponse]): A list of PromptRequestResponse objects representing the
            conversation to be scored. This can contain one PromptRequestResponse object if you are just
            scoring individual assistant responses.
        human_scores (List): A list of human-assigned scores for the responses. Each entry in the list corresponds to
            a different person's score for the same response/conversation.
    """

    conversation: List[PromptRequestResponse]
    human_scores: List


@dataclass
class HarmHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for a specific harm category. This class includes the
    PromptRequestResponses and a list of human scores, which are floats between 0.0 and 1.0 inclusive,
    representing the degree of harm severity where 0.0 is minimal and 1.0 is maximal. The harm category is a
    string that represents the type of harm (e.g., "hate_speech", "misinformation", etc.).
    """

    human_scores: List[float]
    # For now, this is a string, but may be enum or Literal in the future.
    harm_category: str

    def __post_init__(self):
        if not all(score >= 0.0 and score <= 1.0 for score in self.human_scores):
            raise ValueError("All human scores must be between 0.0 and 1.0 inclusive.")


@dataclass
class ObjectiveHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for a specific objective. This class includes the
    PromptRequestResponses and a list of human scores, which are booleans indicating whether the response/conversation
    meets the objective (e.g., 0 for not meeting the objective, 1 for meeting the objective). The objective is a
    string that represents the objective (e.g., "how to make a Molotov cocktail?).
    """

    human_scores: List[bool]
    objective: str


class HumanLabeledDataset:
    """
    A class that represents a human-labeled dataset, including the entries and each of their corresponding
    human scores. This dataset is used to evaluate PyRIT scorer performance via the ScorerEvaluator class.
    HumanLabeledDatasets can be constructed from a CSV file.

    Args:
        name (str): The name of the human-labeled dataset. For datasets of uniform type, this is often the harm
            category (e.g. hate_speech) or objective. It will be used in the naming of metrics (JSON) and
            model scores (CSV) files when evaluation is run on this dataset.
        entries (List[HumanLabeledEntry]): A list of HumanLabeledEntry objects representing the entries in the dataset.
        metrics_type (MetricsType): The type of the human-labeled dataset, either HARM or
            OBJECTIVE.
    """

    def __init__(
        self,
        *,
        name: str,
        entries: List[HumanLabeledEntry],
        metrics_type: MetricsType,
    ):
        if not name:
            raise ValueError("Dataset name cannot be an empty string.")

        self.name = name
        self.entries = entries
        self.metrics_type = metrics_type

        for entry in self.entries:
            self._validate_entry(entry)

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: Union[str, Path],
        metrics_type: MetricsType,
        assistant_response_col_name: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        assistant_response_data_type_col_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> "HumanLabeledDataset":
        """
        Load a human-labeled dataset from a CSV file. This only allows for single turn scored text responses.

        Args:
            csv_path (Union[str, Path]): The path to the CSV file.
            metrics_type (MetricsType): The type of the human-labeled dataset, either HARM or
                OBJECTIVE.
            assistant_response_col_name (str): The name of the column containing the assistant responses.
            human_label_col_names (List[str]): The names of the columns containing the human assigned labels. For
                harm datasets, the CSV file should contain float scores between 0.0 and 1.0 for each response.
                For objective datasets, the CSV file should contain a 0 or 1 for each response.
            objective_or_harm_col_name (str): The name of the column containing the objective or harm category for
                each response.
            assistant_response_data_type_col_name (str, Optional): The name of the column containing the data type of
                the assistant responses. If not specified, it is assumed that the responses are text.
            dataset_name: (str, Optional): The name of the dataset. If not provided, it will be inferred from the CSV
                file name.

        Returns:
            HumanLabeledDataset: The human-labeled dataset object.
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file does not exist: {csv_path}")

        eval_df = pd.read_csv(csv_path)
        # cls._validate_fields
        cls._validate_columns(
            eval_df=eval_df,
            human_label_col_names=human_label_col_names,
            assistant_response_col_name=assistant_response_col_name,
            objective_or_harm_col_name=objective_or_harm_col_name,
            assistant_response_data_type_col_name=assistant_response_data_type_col_name,
        )

        responses_to_score = eval_df[assistant_response_col_name].tolist()
        all_human_scores = eval_df[human_label_col_names].values.tolist()
        objectives_or_harms = eval_df[objective_or_harm_col_name].tolist()
        if assistant_response_data_type_col_name:
            data_types = eval_df[assistant_response_data_type_col_name].tolist()
        else:
            data_types = ["text"] * len(eval_df[assistant_response_col_name])

        entries: List[HumanLabeledEntry] = []
        for response_to_score, human_scores, objective_or_harm, data_type in zip(
            responses_to_score, all_human_scores, objectives_or_harms, data_types
        ):
            cls._validate_fields(
                response_to_score=response_to_score,
                human_scores=human_scores,
                objective_or_harm=objective_or_harm,
                data_type=data_type,
            )
            response_to_score = str(response_to_score).strip()
            objective_or_harm = str(objective_or_harm).strip()
            data_type = str(data_type).strip()

            # Each list of request_responses consists only of a single assistant response since each row
            # is treated as a single turn conversation.
            request_responses = [
                PromptRequestResponse(
                    request_pieces=[
                        PromptRequestPiece(
                            role="assistant", 
                            original_value=response_to_score, 
                            original_value_data_type=cast(PromptDataType, data_type)
                        )
                    ],
                )
            ]
            if metrics_type == MetricsType.HARM:
                entry = cls._construct_harm_entry(
                    request_responses=request_responses,
                    harm=objective_or_harm,
                    human_scores=human_scores,
                )
            else:
                entry = cls._construct_objective_entry(
                    request_responses=request_responses,
                    objective=objective_or_harm,
                    human_scores=human_scores,
                )
            entries.append(entry)

        dataset_name = dataset_name or Path(csv_path).stem
        return cls(entries=entries, name=dataset_name, metrics_type=metrics_type)

    def add_entries(self, entries: List[HumanLabeledEntry]):
        """
        Add multiple entries to the human-labeled dataset.

        Args:
            entries (List[HumanLabeledEntry]): A list of entries to add.
        """
        for entry in entries:
            self.add_entry(entry)

    def add_entry(self, entry: HumanLabeledEntry):
        """
        Add a new entry to the human-labeled dataset.

        Args:
            entry (HumanLabeledEntry): The entry to add.
        """
        self._validate_entry(entry)
        self.entries.append(entry)

    def _validate_entry(self, entry: HumanLabeledEntry):
        if self.metrics_type == MetricsType.HARM:
            if not isinstance(entry, HarmHumanLabeledEntry):
                raise ValueError("All entries must be HarmHumanLabeledEntry instances for harm datasets.")
            if self.entries:
                first_entry = self.entries[0]
                # if statement for static type checking
                if isinstance(first_entry, HarmHumanLabeledEntry):
                    if entry.harm_category != first_entry.harm_category:
                        logger.warning(
                            "All entries in a harm dataset should have the same harm category. "
                            "Evaluating a dataset with multiple harm categories is not currently supported."
                        )
        elif self.metrics_type == MetricsType.OBJECTIVE:
            if not isinstance(entry, ObjectiveHumanLabeledEntry):
                raise ValueError("All entries must be ObjectiveHumanLabeledEntry instances for objective datasets.")

    @staticmethod
    def _validate_columns(
        *,
        eval_df: pd.DataFrame,
        human_label_col_names: List[str],
        assistant_response_col_name: str,
        objective_or_harm_col_name: str,
        assistant_response_data_type_col_name: Optional[str] = None,
    ):
        """
        Validate that the required columns exist in the DataFrame (representing the human-labeled dataset)
        and that they are of the correct length and do not contain NaN values.

        Args:
            eval_df (pd.DataFrame): The DataFrame to validate.
            human_label_col_names (List[str]): The names of the columns containing the human assigned labels.
            assistant_response_col_name (str): The name of the column containing the assistant responses.
            objective_or_harm_col_name (str): The name of the column containing the objective or harm category
                for each response.
            assistant_response_data_type_col_name (Optional[str]): The name of the column containing the data type
                of the assistant responses.
        """
        if len(eval_df.columns) != len(set(eval_df.columns)):
            raise ValueError("Column names in the dataset must be unique.")

        required_columns = human_label_col_names + [assistant_response_col_name, objective_or_harm_col_name]
        if assistant_response_data_type_col_name:
            required_columns.append(assistant_response_data_type_col_name)

        for column in required_columns:
            if column not in eval_df.columns:
                raise ValueError(f"Column {column} is missing from the dataset.")
            if eval_df[column].isna().any():
                raise ValueError(f"Column {column} contains NaN values.")

    @staticmethod
    def _validate_fields(
        *,
        response_to_score,
        human_scores: List,
        objective_or_harm,
        data_type,
    ):
        """
        Validate the fields needed for a human-labeled dataset entry.

        Args:
            response_to_score: The response to score.
            human_scores (List): The human scores for the response.
            objective_or_harm: The objective or harm category for the response.
            data_type: The data type of the response (e.g., "text", "image", etc.).
        """
        if not response_to_score or not str(response_to_score).strip():
            raise ValueError("One or more of the responses is empty. Ensure that the file contains " "valid responses.")
        if not all(isinstance(score, (int, float)) for score in human_scores):
            raise ValueError(
                "Human scores must be a list of numeric values (int or float). Ensure that the file contains valid"
                " human scores. True and False values should be represented as 1 and 0 respectively."
            )
        if not objective_or_harm or not isinstance(objective_or_harm, str) or not str(objective_or_harm).strip():
            raise ValueError(
                "An objective or harm category is missing or not a string. Ensure that the file contains"
                " valid objectives or harm categories."
            )
        if not data_type or not isinstance(data_type, str) or data_type.strip() not in get_args(PromptDataType):
            raise ValueError(f"One of the data types is invalid. Valid types are: {get_args(PromptDataType)}.")

    @staticmethod
    def _construct_harm_entry(*, request_responses: List[PromptRequestResponse], harm: str, human_scores: List):
        float_scores = [float(score) for score in human_scores]
        return HarmHumanLabeledEntry(request_responses, float_scores, harm)

    @staticmethod
    def _construct_objective_entry(
        *, request_responses: List[PromptRequestResponse], objective: str, human_scores: List
    ):
        # Convert scores to int before casting to bool in case the values (0, 1) are parsed as strings
        bool_scores = [bool(int(score)) for score in human_scores]
        return ObjectiveHumanLabeledEntry(request_responses, bool_scores, objective)
