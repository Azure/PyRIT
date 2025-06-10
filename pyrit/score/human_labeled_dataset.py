# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd

from pyrit.models import PromptDataType, PromptRequestPiece, PromptRequestResponse

logger = logging.getLogger(__name__)


@dataclass
class HumanLabeledEntry:
    """
    A class that represents a human-labeled dataset entry, including the PromptRequestResponses and list of human scores.

    responses_to_score: List[PromptRequestResponse]
        A list of PromptRequestResponse objects representing the responses to be scored. This is represented as a list
        to accommodate multiple turns in a conversation.
    """

    responses_to_score: List[PromptRequestResponse]
    human_scores: List


@dataclass
class HarmHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for harmful content, including the PromptRequestResponses and list of human scores.
    """

    human_scores: List[float]
    # For now, this is a string, but should be enum or Literal in the future.
    harm_category: str

    def __post_init__(self):
        if not all(score >= 0.0 and score <= 1.0 for score in self.human_scores):
            raise ValueError("All human scores must be between 0.0 and 1.0 inclusive.")


@dataclass
class ObjectiveHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for objectives, including the PromptRequestResponses and list of human scores.
    """

    human_scores: List[bool]
    objective: str


class HumanLabeledDataset:
    """
    A class that represents a human-labeled dataset, including the entries and their corresponding human scores.
    If harm_category or objective is specified, all entries must have the same value for that field. If not specified,
    all entries can have different values for that field.

    Parameters:
        name (str): The name of the human-labeled dataset. For datasets of uniform type, this is often the harm 
            category (e.g. hate_speech) or objective. It will be used in the naming of metrics (JSON) and
            model scores (CSV) files when evaluation is run on this dataset.
        entries (List[HumanLabeledEntry]): A list of HumanLabeledEntry objects representing the entries in the dataset.
        type (Literal["harm", "objective"]): The type of the human-labeled dataset, either "harm" or "objective".
    """

    def __init__(
        self,
        *,
        name: str,
        entries: List[HumanLabeledEntry],
        type: Literal["harm", "objective"],
    ):
        self.name = name
        self.entries = entries
        self.type = type
        if self.type == "harm":
            if not all(isinstance(entry, HarmHumanLabeledEntry) for entry in self.entries):
                raise ValueError("All entries must be HarmHumanLabeledEntry instances for harm datasets.")
            if len(set(entry.harm_category for entry in self.entries)) > 1:
                logger.warning("All entries in a harm dataset should have the same harm category."
                               "Evaluating a dataset with multiple harm categories is not currently supported.")
        elif self.type == "objective":
            if not all(isinstance(entry, ObjectiveHumanLabeledEntry) for entry in self.entries):
                raise ValueError("All entries must be ObjectiveHumanLabeledEntry instances for objective datasets.")
            

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        dataset_name: str,
        type: Literal["harm", "objective"],
        assistant_responses_col_name: str,
        human_label_col_names: List[str],
        objective_or_harm_col_name: str,
        assistant_responses_data_type_col_name: Optional[PromptDataType] = None,
    ) -> "HumanLabeledDataset":
        """
        Load a human-labeled dataset from a CSV file. This only allows for single turn scored text responses.

        Args:
            csv_path (str): The path to the CSV file.
            dataset_name (str): The name of the human-labeled dataset.
            type (Literal["harm", "objective"]): The type of the human-labeled dataset.
            assistant_responses_col_name (str): The name of the column containing the assistant responses.
            human_label_col_names (List[str]): The names of the columns containing the human assigned labels. For 
                harm datasets, the CSV file should contain float scores between 0.0 and 1.0 for each response. 
                For objective datasets, the CSV file should contain a 0 or 1 for each response.
            objective_or_harm_col_name (Optional[str]): The name of the column containing the objective or harm category.
            assistant_responses_data_type_col_name (Optional[str]): The name of the column containing the data type of 
                the assistant responses. If not specified, it is assumed that the responses are text.

        Returns:
            HumanLabeledDataset: The loaded human-labeled dataset.
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file does not exist: {csv_path}")

        eval_df = pd.read_csv(csv_path)
        required_columns = set(human_label_col_names + [assistant_responses_col_name, objective_or_harm_col_name])
        if assistant_responses_data_type_col_name:
            required_columns.add(assistant_responses_data_type_col_name)
        assert required_columns.issubset(eval_df.columns), "Missing required columns in the dataset"
        for human_label_col in human_label_col_names:
            assert len(eval_df[human_label_col]) == len(
                eval_df[assistant_responses_col_name]
            ), f"Number of scores in column {human_label_col} does not match the number of responses"

        assert len(eval_df[objective_or_harm_col_name]) == len(
            eval_df[assistant_responses_col_name]
        ), f"Number of entries in column {objective_or_harm_col_name} does not match the number of responses"

        if assistant_responses_data_type_col_name:
            assert len(eval_df[assistant_responses_data_type_col_name]) == len(
                eval_df[assistant_responses_col_name]
            ), f"Number of entries in column {assistant_responses_data_type_col_name} does not match the number of responses"
            data_types = eval_df[assistant_responses_data_type_col_name].tolist()
        else:
            data_types = ["text"] * len(eval_df[assistant_responses_col_name])
        responses_to_score = eval_df[assistant_responses_col_name].tolist()
        all_human_scores = eval_df[human_label_col_names].values.tolist()
        objectives_or_harms = eval_df[objective_or_harm_col_name].tolist()

        entries: List[HumanLabeledEntry] = []
        for response_to_score, human_scores, objective_or_harm, data_type in zip(
            responses_to_score, all_human_scores, objectives_or_harms, data_types
        ):
            if not response_to_score or response_to_score != response_to_score:
                raise ValueError("Response to score is empty or NaN. Ensure that the file contains valid responses.")
            # Each list of request_responses consists only of a single assistant response since each row
            # is treated as a single turn conversation.
            request_responses = [
                PromptRequestResponse(
                    request_pieces=[PromptRequestPiece(role="assistant", 
                                                       original_value=response_to_score, 
                                                       original_value_data_type=data_type)],
                )
            ]
            if type == "harm":
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
        return cls(
            entries=entries,
            name=dataset_name,
            type=type,
        )

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
        if self.type == "harm":
            if not isinstance(entry, HarmHumanLabeledEntry):
                raise ValueError("Entry must be a HarmHumanLabeledEntry for harm datasets.")
            if self.entries and entry.harm_category != self.entries[0].harm_category:
                logger.warning("All entries in a harm dataset should have the same harm category."
                               "Evaluating a dataset with multiple harm categories is not currently supported.")
        if self.type == "objective" and not isinstance(entry, ObjectiveHumanLabeledEntry):
            raise ValueError("Entry must be an ObjectiveHumanLabeledEntry for objective datasets.")
        self.entries.append(entry)

    @staticmethod
    def validate_columns():
        pass

    @staticmethod
    def _construct_harm_entry(request_responses: List[PromptRequestResponse], harm: str, human_scores: List):
        if not harm or harm!=harm:
            raise ValueError("Harm category is missing or NaN. Ensure that each response has a valid harm category.")
        float_scores = [float(score) for score in human_scores]
        return HarmHumanLabeledEntry(request_responses, float_scores, harm)

    @staticmethod
    def _construct_objective_entry(request_responses: List[PromptRequestResponse], objective: str, human_scores: List):
        if not objective or objective!=objective:
            raise ValueError("Objective is missing or NaN. Ensure that each response has a valid objective.")
        # Convert scores to int before casting to bool in case the values (0, 1) are parsed as strings
        bool_scores = [bool(int(score)) for score in human_scores]
        return ObjectiveHumanLabeledEntry(request_responses, bool_scores, objective)