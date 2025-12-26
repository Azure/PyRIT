# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, cast, get_args

import pandas as pd

from pyrit.common.utils import verify_and_resolve_path
from pyrit.models import Message, MessagePiece, PromptDataType
from pyrit.score import MetricsType

logger = logging.getLogger(__name__)

# Standard column names for evaluation datasets
STANDARD_HUMAN_LABEL_COL = "human_score"
STANDARD_OBJECTIVE_COL = "objective"
STANDARD_HARM_COL = "harm_category"
STANDARD_ASSISTANT_RESPONSE_COL = "assistant_response"
STANDARD_DATA_TYPE_COL = "data_type"


@dataclass
class HumanLabeledEntry:
    """
    A class that represents an entry in a dataset of assistant responses that have been scored by humans. It is used
    to evaluate PyRIT scorer performance as measured by degree of alignment with human labels. This class includes
    the Messages and a list of human-assigned scores, which are floats between 0.0 and 1.0 inclusive
    (representing degree of severity) for harm datasets, and booleans for objective datasets.

    Parameters:
        conversation (List[Message]): A list of Message objects representing the
            conversation to be scored. This can contain one Message object if you are just
            scoring individual assistant responses.
        human_scores (List): A list of human-assigned scores for the responses. Each entry in the list corresponds to
            a different person's score for the same response/conversation.
    """

    conversation: List[Message]
    human_scores: List

    def __post_init__(self) -> None:
        """
        Validate that conversation and human_scores are not None and have positive lengths.

        Raises:
            ValueError: If conversation or human_scores is None or empty.
        """
        if self.conversation is None or len(self.conversation) == 0:
            raise ValueError("conversation must not be None or empty.")
        if self.human_scores is None or len(self.human_scores) == 0:
            raise ValueError("human_scores must not be None or empty.")


@dataclass
class HarmHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for a specific harm category. This class includes the
    Messages and a list of human scores, which are floats between 0.0 and 1.0 inclusive,
    representing the degree of harm severity where 0.0 is minimal and 1.0 is maximal. The harm category is a
    string that represents the type of harm (e.g., "hate_speech", "misinformation", etc.).
    """

    human_scores: List[float]
    # For now, this is a string, but may be enum or Literal in the future.
    harm_category: str

    def __post_init__(self) -> None:
        """
        Validate harm category and human scores.

        Raises:
            ValueError: If harm_category is None or empty, or if any human score is not between 0.0 and 1.0 inclusive.
        """
        super().__post_init__()
        
        if not self.harm_category or not self.harm_category.strip():
            raise ValueError("harm_category must not be None or empty.")
        
        if not all(score >= 0.0 and score <= 1.0 for score in self.human_scores):
            raise ValueError("All human scores must be between 0.0 and 1.0 inclusive.")


@dataclass
class ObjectiveHumanLabeledEntry(HumanLabeledEntry):
    """
    A class that represents a human-labeled dataset entry for a specific objective. This class includes the
    Messages and a list of human scores, which are booleans indicating whether the response/conversation
    meets the objective (e.g., 0 for not meeting the objective, 1 for meeting the objective). The objective is a
    string that represents the objective (e.g., "how to make a Molotov cocktail?).
    """

    human_scores: List[bool]
    objective: str

    def __post_init__(self) -> None:
        """
        Validate objective field.

        Raises:
            ValueError: If objective is None or empty.
        """
        super().__post_init__()
        
        if not self.objective or not self.objective.strip():
            raise ValueError("objective must not be None or empty.")


class HumanLabeledDataset:
    """
    A class that represents a human-labeled dataset, including the entries and each of their corresponding
    human scores. This dataset is used to evaluate PyRIT scorer performance via the ScorerEvaluator class.
    HumanLabeledDatasets can be constructed from a CSV file.
    """

    def __init__(self, *, name: str, entries: List[HumanLabeledEntry], metrics_type: MetricsType, version: str):
        """
        Initialize the HumanLabeledDataset.

        Args:
            name (str): The name of the human-labeled dataset. For datasets of uniform type, this is often the harm
                category (e.g. hate_speech) or objective. It will be used in the naming of metrics (JSON) and
                model scores (CSV) files when evaluation is run on this dataset.
            entries (List[HumanLabeledEntry]): A list of entries in the dataset.
            metrics_type (MetricsType): The type of the human-labeled dataset, either HARM or
                OBJECTIVE.
            version (str): The version of the human-labeled dataset.

        Raises:
            ValueError: If the dataset name is an empty string.
        """
        if not name:
            raise ValueError("Dataset name cannot be an empty string.")

        self.name = name
        self.entries = entries
        self.metrics_type = metrics_type
        self.version = version

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: Union[str, Path],
        metrics_type: MetricsType,
        dataset_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "HumanLabeledDataset":
        """
        Load a human-labeled dataset from a CSV file with standard column names.
        
        Expected CSV format:
        - 'assistant_response': The assistant's response text
        - 'human_score': Human-assigned label (can have multiple columns for multiple raters)
        - 'objective': For OBJECTIVE datasets, the objective being evaluated
        - 'data_type': Optional data type (defaults to 'text' if not present)
        
        You can optionally include a # comment line at the top of the CSV file to specify 
        the dataset version (using # version=x.y).

        Args:
            csv_path (Union[str, Path]): The path to the CSV file.
            metrics_type (MetricsType): The type of the human-labeled dataset, either HARM or OBJECTIVE.
            dataset_name (str, Optional): The name of the dataset. If not provided, it will be inferred 
                from the CSV file name.
            version (str, Optional): The version of the dataset. If not provided here, it will be inferred 
                from the CSV file if a version comment line "#version=" is present.

        Returns:
            HumanLabeledDataset: The human-labeled dataset object.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If version is not provided and not found in the CSV file via comment line "# version=".
        """
        csv_path = verify_and_resolve_path(csv_path)
        # Read the first line to check for version info
        if not version:
            with open(csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#") and "version=" in first_line:
                    # Extract version, handling trailing commas from CSV format (e.g., "# version=1.0,,,,")
                    version_part = first_line.split("=", 1)[1].strip()
                    version = version_part.split(",")[0].strip()
                else:
                    raise ValueError("Version not specified and not found in CSV file.")

        # Try UTF-8 first, fall back to latin-1 for files with special characters
        try:
            eval_df = pd.read_csv(csv_path, comment="#", encoding="utf-8")
        except UnicodeDecodeError:
            eval_df = pd.read_csv(csv_path, comment="#", encoding="latin-1")

        # Validate required columns exist and have no NaN values
        cls._validate_csv_columns(eval_df=eval_df, metrics_type=metrics_type)

        # Determine human label columns (all columns starting with standard prefix)
        human_label_col_names = [col for col in eval_df.columns if col.startswith(STANDARD_HUMAN_LABEL_COL)]
        if not human_label_col_names:
            raise ValueError(
                f"No human score columns found. Expected columns starting with '{STANDARD_HUMAN_LABEL_COL}'."
            )

        # Get data type column if it exists, otherwise default to 'text'
        has_data_type_col = STANDARD_DATA_TYPE_COL in eval_df.columns

        responses_to_score = eval_df[STANDARD_ASSISTANT_RESPONSE_COL].tolist()
        all_human_scores = eval_df[human_label_col_names].values.tolist()
        # Use appropriate column based on metrics type
        objective_or_harm_col = STANDARD_HARM_COL if metrics_type == MetricsType.HARM else STANDARD_OBJECTIVE_COL
        objectives_or_harms = eval_df[objective_or_harm_col].tolist()
        if has_data_type_col:
            data_types = eval_df[STANDARD_DATA_TYPE_COL].tolist()
        else:
            data_types = ["text"] * len(eval_df[STANDARD_ASSISTANT_RESPONSE_COL])

        entries: List[HumanLabeledEntry] = []
        for response_to_score, human_scores, objective_or_harm, data_type in zip(
            responses_to_score, all_human_scores, objectives_or_harms, data_types
        ):
            response_to_score = str(response_to_score).strip()
            objective_or_harm = str(objective_or_harm).strip()
            data_type = str(data_type).strip()

            # Each list of messages consists only of a single assistant response since each row
            # is treated as a single turn conversation.
            messages = [
                Message(
                    message_pieces=[
                        MessagePiece(
                            role="assistant",
                            original_value=response_to_score,
                            original_value_data_type=cast(PromptDataType, data_type),
                        )
                    ],
                )
            ]
            if metrics_type == MetricsType.HARM:
                entry = cls._construct_harm_entry(
                    messages=messages,
                    harm=objective_or_harm,
                    human_scores=human_scores,
                )
            else:
                entry = cls._construct_objective_entry(
                    messages=messages,
                    objective=objective_or_harm,
                    human_scores=human_scores,
                )
            entries.append(entry)

        dataset_name = dataset_name or Path(csv_path).stem
        return cls(entries=entries, name=dataset_name, metrics_type=metrics_type, version=version)

    def validate(self) -> None:
        """
        Validate that the dataset is internally consistent.
        
        Checks that all entries match the dataset's metrics_type and, for HARM datasets,
        that all entries have the same harm_category.
        
        Raises:
            ValueError: If entries don't match metrics_type or harm categories are inconsistent.
        """
        if not self.entries:
            return
        
        if self.metrics_type == MetricsType.HARM:
            harm_categories = set()
            for index, entry in enumerate(self.entries):
                if not isinstance(entry, HarmHumanLabeledEntry):
                    raise ValueError(
                        f"Entry at index {index} is not a HarmHumanLabeledEntry, "
                        "but the HumanLabeledDataset type is HARM."
                    )
                harm_categories.add(entry.harm_category)
            
            if len(harm_categories) > 1:
                raise ValueError("Evaluating a dataset with multiple harm categories is not currently supported.")
        
        elif self.metrics_type == MetricsType.OBJECTIVE:
            for index, entry in enumerate(self.entries):
                if not isinstance(entry, ObjectiveHumanLabeledEntry):
                    raise ValueError(
                        f"Entry at index {index} is not an ObjectiveHumanLabeledEntry, "
                        "but the HumanLabeledDataset type is OBJECTIVE."
                    )

    @classmethod
    def _validate_csv_columns(cls, *, eval_df: pd.DataFrame, metrics_type: MetricsType) -> None:
        """
        Validate that the required standard columns exist in the DataFrame.

        Args:
            eval_df (pd.DataFrame): The DataFrame to validate.
            metrics_type (MetricsType): The type of dataset (HARM or OBJECTIVE).

        Raises:
            ValueError: If any required column is missing or if column names are not unique.
        """
        if len(eval_df.columns) != len(set(eval_df.columns)):
            raise ValueError("Column names in the dataset must be unique.")

        # Required columns depend on metrics type
        objective_or_harm_col = STANDARD_HARM_COL if metrics_type == MetricsType.HARM else STANDARD_OBJECTIVE_COL
        required_columns = [STANDARD_ASSISTANT_RESPONSE_COL, objective_or_harm_col]
        
        for column in required_columns:
            if column not in eval_df.columns:
                raise ValueError(
                    f"Required column '{column}' is missing from the dataset. "
                    f"Found columns: {list(eval_df.columns)}"
                )
            if eval_df[column].isna().any():
                raise ValueError(f"Column '{column}' contains NaN values.")
        
        # Check for at least one human score column
        human_score_cols = [col for col in eval_df.columns if col.startswith(STANDARD_HUMAN_LABEL_COL)]
        if not human_score_cols:
            raise ValueError(
                f"No human score columns found. Expected at least one column starting with '{STANDARD_HUMAN_LABEL_COL}'."
            )
        
        # Validate human score columns don't have NaN
        for col in human_score_cols:
            if eval_df[col].isna().any():
                raise ValueError(f"Human score column '{col}' contains NaN values.")

    @staticmethod
    def _construct_harm_entry(*, messages: List[Message], harm: str, human_scores: List):
        float_scores = [float(score) for score in human_scores]
        return HarmHumanLabeledEntry(messages, float_scores, harm)

    @staticmethod
    def _construct_objective_entry(*, messages: List[Message], objective: str, human_scores: List):
        # Convert scores to int before casting to bool in case the values (0, 1) are parsed as strings
        bool_scores = [bool(int(score)) for score in human_scores]
        return ObjectiveHumanLabeledEntry(messages, bool_scores, objective)
