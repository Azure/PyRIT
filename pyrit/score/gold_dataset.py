# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd

from pyrit.models import PromptRequestPiece, PromptRequestResponse


@dataclass
class GoldDatasetEntry:
    """
    A class that represents a gold dataset entry, including the PromptRequestResponses and "gold-label" human scores.

    responses_to_score: List[PromptRequestResponse]
        A list of PromptRequestResponse objects representing the responses to be scored. This is represented as a list
        to accommodate multiple turns in a conversation.
    """

    responses_to_score: List[PromptRequestResponse]
    human_scores: List


@dataclass
class HarmGoldDatasetEntry(GoldDatasetEntry):
    """
    A class that represents a gold dataset entry for harmful content, including the PromptRequestResponses and "gold-label" human scores.
    """

    human_scores: List[float]
    # For now, this is a string, but should be enum or Literal in the future.
    harm_category: str


@dataclass
class ObjectiveGoldDatasetEntry(GoldDatasetEntry):
    """
    A class that represents a gold dataset entry for objectives, including the PromptRequestResponses and "gold-label" human scores.
    """

    human_scores: List[bool]
    objective: str


class GoldDataset:
    """
    A class that represents a gold dataset, including the entries and their corresponding human scores.
    If harm_category or objective is specified, all entries must have the same value for that field. If not specified,
    all entries can have different values for that field.
    """

    entries: List[GoldDatasetEntry]
    type: Literal["harm", "objective"]
    policy_path: Union[str, Path]

    def __init__(
        self,
        *,
        entries: List[GoldDatasetEntry],
        type: Literal["harm", "objective"],
        harm_category: Optional[str] = None,
        objective: Optional[str] = None,
    ):
        self.entries = entries
        self.type = type
        if harm_category and objective:
            raise ValueError("Cannot specify both harm and objective for a gold dataset.")

        if self.type == "harm":
            if not all(isinstance(entry, HarmGoldDatasetEntry) for entry in self.entries):
                raise ValueError("All entries must be HarmGoldDatasetEntry instances for harm datasets.")
            if objective:
                raise ValueError("Objective cannot be specified for a harm dataset.")
            if not all(entry.harm_category == harm_category for entry in self.entries):
                raise ValueError("All entries must have the same harm category if specified.")
            # store top level harm category if provided
            self.harm_category = harm_category
        elif self.type == "objective":
            if not all(isinstance(entry, ObjectiveGoldDatasetEntry) for entry in self.entries):
                raise ValueError("All entries must be ObjectiveGoldDatasetEntry instances for objective datasets.")
            if harm_category:
                raise ValueError("Harm cannot be specified for an objective dataset.")
            if not all(entry.objective == objective for entry in self.entries):
                raise ValueError("All entries must have the same objective if specified.")
            # store top level objective if provided
            self.objective = objective

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        type: Literal["harm", "objective"],
        assistant_response_col: str,
        gold_label_col_names: List[str],
        objective_or_harm_col_name: Optional[str] = None,
        top_level_objective: Optional[str] = None,
        top_level_harm: Optional[str] = None,
    ) -> "GoldDataset":
        """
        Load a gold dataset from a CSV file. This only allows for single turn scored text responses.

        Args:
            csv_path (str): The path to the CSV file.
            type (Literal["harm", "objective"]): The type of the gold dataset.

        Returns:
            GoldDataset: The loaded gold dataset.
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file does not exist: {csv_path}")
        # if not os.path.exists(policy_path):
        #     raise ValueError(f"Policy path does not exist: {policy_path}")
        if type == "harm" and top_level_objective:
            raise ValueError("Top level objective cannot be specified for a harm dataset.")
        if type == "objective" and top_level_harm:
            raise ValueError("Top level harm cannot be specified for an objective dataset.")
        if objective_or_harm_col_name:
            if top_level_harm or top_level_objective:
                raise ValueError(
                    "Top level harm or objective cannot be specified if objective_or_harm_col_name is provided."
                )

        eval_df = pd.read_csv(csv_path)
        required_columns = set(gold_label_col_names + [assistant_response_col])
        if objective_or_harm_col_name:
            required_columns.add(objective_or_harm_col_name)
        assert required_columns.issubset(eval_df.columns), "Missing required columns in the dataset"
        for gold_label_col in gold_label_col_names:
            assert len(eval_df[gold_label_col]) == len(
                eval_df[assistant_response_col]
            ), f"Number of scores in column {gold_label_col} does not match the number of responses"

        objectives_or_harms = [None] * len(eval_df[assistant_response_col])
        if objective_or_harm_col_name:
            assert len(eval_df[objective_or_harm_col_name]) == len(
                eval_df[assistant_response_col]
            ), f"Number of entries in column {objective_or_harm_col_name} does not match the number of responses"
            objectives_or_harms = eval_df[objective_or_harm_col_name].tolist()
        responses_to_score = eval_df[assistant_response_col].tolist()
        gold_label_scores = eval_df[gold_label_col_names].values.tolist()

        entries: List[GoldDatasetEntry] = []
        for response_to_score, gold_label_score, objective_or_harm in zip(
            responses_to_score, gold_label_scores, objectives_or_harms
        ):
            # Each list of request_responses consists only of a single assistant response since each row
            # is treated as a single turn conversation.
            request_responses = [
                PromptRequestResponse(
                    request_pieces=[PromptRequestPiece(role="assistant", original_value=response_to_score)]
                )
            ]
            if type == "harm":
                float_scores = [float(score) for score in gold_label_score]
                harm = top_level_harm if top_level_harm else objective_or_harm
                if not harm:
                    raise ValueError(
                        "Harm category must be specified either in the data or at top-level for harm datasets."
                        f"Response: {response_to_score} does not have a harm category."
                    )
                entries.append(HarmGoldDatasetEntry(request_responses, float_scores, harm))
            elif type == "objective":
                # Convert scores to int before casting to bool in case the values (0, 1) are parsed as strings
                bool_scores = [bool(int(score)) for score in gold_label_score]
                objective = top_level_objective if top_level_objective else objective_or_harm
                if not objective:
                    raise ValueError(
                        "Objective must be specified either in the data or at top-level for objective datasets. "
                        f"Response: {response_to_score} does not have an objective."
                    )
                entries.append(ObjectiveGoldDatasetEntry(request_responses, bool_scores, objective))

        return cls(
            entries=entries,
            type=type,
            harm_category=top_level_harm if type == "harm" else None,
            objective=top_level_objective if type == "objective" else None,
        )

    def add_entry(self, entry: GoldDatasetEntry):
        """
        Add a new entry to the gold dataset.

        Args:
            entry (GoldDatasetEntry): The entry to add.
        """
        if self.type == "harm" and not isinstance(entry, HarmGoldDatasetEntry):
            raise ValueError("Entry must be a HarmGoldDatasetEntry for harm datasets.")
        if self.type == "objective" and not isinstance(entry, ObjectiveGoldDatasetEntry):
            raise ValueError("Entry must be an ObjectiveGoldDatasetEntry for objective datasets.")
        self.entries.append(entry)
