# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for validating CSV schema compliance in scorer evaluation datasets.

These tests ensure all CSV files in the scorer_evals directory follow the
standardized column naming conventions defined in scorer_evaluator.py.
"""

import csv
from pathlib import Path
from typing import List, Set

import pytest

from pyrit.common.path import SCORER_EVALS_HARM_PATH, SCORER_EVALS_OBJECTIVE_PATH, SCORER_EVALS_REFUSAL_SCORER_PATH
from pyrit.score.scorer_evaluation.scorer_evaluator import (
    STANDARD_ASSISTANT_RESPONSE_COL,
    STANDARD_DATA_TYPE_COL,
    STANDARD_HARM_COL,
    STANDARD_HUMAN_LABEL_COL,
    STANDARD_OBJECTIVE_COL,
)


class TestObjectiveScorerEvalCSVSchema:
    """Test that all objective scorer evaluation CSVs have the correct schema."""

    @pytest.fixture(scope="class")
    def objective_csv_files(self) -> List[Path]:
        """Get all CSV files in the objective scorer evals directory."""
        return list(Path(SCORER_EVALS_OBJECTIVE_PATH).glob("*.csv"))

    def test_objective_csv_files_exist(self, objective_csv_files: List[Path]) -> None:
        """Verify that objective CSV files exist."""
        assert len(objective_csv_files) > 0, "No objective CSV files found"

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_OBJECTIVE_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_objective_csv_has_required_columns(self, csv_file: Path) -> None:
        """
        Test that each objective CSV has all required columns.
        
        Required columns for objective scorer evaluation:
        - objective: The objective being evaluated
        - assistant_response: The model's response
        - human_score: The human-labeled ground truth score
        - data_type: The type of data (e.g., "text")
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                # Reset to beginning if no version line
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            required_columns = {
                STANDARD_OBJECTIVE_COL,
                STANDARD_ASSISTANT_RESPONSE_COL,
                STANDARD_HUMAN_LABEL_COL,
                STANDARD_DATA_TYPE_COL,
            }
            
            missing_columns = required_columns - columns
            assert not missing_columns, (
                f"CSV {csv_file.name} is missing required columns: {missing_columns}. "
                f"Found columns: {columns}"
            )

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_OBJECTIVE_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_objective_csv_column_names_exact(self, csv_file: Path) -> None:
        """
        Test that objective CSVs use only expected column names.
        
        This ensures no typos or legacy column names remain.
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            # Objective CSVs may have harm_category for reference but it's optional
            allowed_columns = {
                STANDARD_OBJECTIVE_COL,
                STANDARD_ASSISTANT_RESPONSE_COL,
                STANDARD_HUMAN_LABEL_COL,
                STANDARD_DATA_TYPE_COL,
                STANDARD_HARM_COL,  # Optional reference column
            }
            
            unexpected_columns = columns - allowed_columns
            assert not unexpected_columns, (
                f"CSV {csv_file.name} has unexpected columns: {unexpected_columns}. "
                f"Allowed columns: {allowed_columns}"
            )


class TestHarmScorerEvalCSVSchema:
    """Test that all harm scorer evaluation CSVs have the correct schema."""

    @pytest.fixture(scope="class")
    def harm_csv_files(self) -> List[Path]:
        """Get all CSV files in the harm scorer evals directory."""
        return list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv"))

    def test_harm_csv_files_exist(self, harm_csv_files: List[Path]) -> None:
        """Verify that harm CSV files exist."""
        assert len(harm_csv_files) > 0, "No harm CSV files found"

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_harm_csv_has_required_columns(self, csv_file: Path) -> None:
        """
        Test that each harm CSV has all required columns.
        
        Required columns for harm scorer evaluation:
        - harm_category: The harm category being evaluated
        - objective: The objective/prompt context
        - assistant_response: The model's response
        - data_type: The type of data (e.g., "text")
        - human_score_1: The primary human-labeled ground truth score
        
        Note: Harm CSVs may have additional human_score_2, human_score_3, etc.
        for multi-annotator datasets.
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            required_columns = {
                STANDARD_HARM_COL,
                STANDARD_OBJECTIVE_COL,
                STANDARD_ASSISTANT_RESPONSE_COL,
                STANDARD_DATA_TYPE_COL,
                "human_score_1",  # Harm CSVs use numbered human scores
            }
            
            missing_columns = required_columns - columns
            assert not missing_columns, (
                f"CSV {csv_file.name} is missing required columns: {missing_columns}. "
                f"Found columns: {columns}"
            )

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_harm_csv_has_human_score_columns(self, csv_file: Path) -> None:
        """
        Test that harm CSVs have at least one human_score_N column.
        
        Harm CSVs support multiple annotators with columns:
        - human_score_1 (required)
        - human_score_2 (optional)
        - human_score_3 (optional)
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            # Find all human_score_* columns
            human_score_cols = {col for col in columns if col.startswith("human_score_")}
            
            assert human_score_cols, (
                f"CSV {csv_file.name} has no human_score_* columns. "
                f"Found columns: {columns}"
            )
            
            # Verify human_score_1 specifically exists
            assert "human_score_1" in human_score_cols, (
                f"CSV {csv_file.name} must have human_score_1 column. "
                f"Found human_score columns: {human_score_cols}"
            )


class TestRefusalScorerEvalCSVSchema:
    """Test that all refusal scorer evaluation CSVs have the correct schema."""

    @pytest.fixture(scope="class")
    def refusal_csv_files(self) -> List[Path]:
        """Get all CSV files in the refusal scorer evals directory."""
        return list(Path(SCORER_EVALS_REFUSAL_SCORER_PATH).glob("*.csv"))

    def test_refusal_csv_files_exist(self, refusal_csv_files: List[Path]) -> None:
        """Verify that refusal CSV files exist."""
        assert len(refusal_csv_files) > 0, "No refusal CSV files found"

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_REFUSAL_SCORER_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_refusal_csv_has_required_columns(self, csv_file: Path) -> None:
        """
        Test that each refusal CSV has all required columns.
        
        Required columns for refusal scorer evaluation:
        - objective: The objective being evaluated
        - assistant_response: The model's response
        - human_score: The human-labeled ground truth score
        - data_type: The type of data (e.g., "text")
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            required_columns = {
                STANDARD_OBJECTIVE_COL,
                STANDARD_ASSISTANT_RESPONSE_COL,
                STANDARD_HUMAN_LABEL_COL,
                STANDARD_DATA_TYPE_COL,
            }
            
            missing_columns = required_columns - columns
            assert not missing_columns, (
                f"CSV {csv_file.name} is missing required columns: {missing_columns}. "
                f"Found columns: {columns}"
            )

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_REFUSAL_SCORER_PATH).glob("*.csv")),
        ids=lambda p: p.name,
    )
    def test_refusal_csv_column_names_exact(self, csv_file: Path) -> None:
        """
        Test that refusal CSVs use only expected column names.
        
        This ensures no typos or legacy column names remain.
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip version line if present
            first_line = f.readline()
            if not first_line.startswith("# dataset_version="):
                f.seek(0)
            
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            
            allowed_columns = {
                STANDARD_OBJECTIVE_COL,
                STANDARD_ASSISTANT_RESPONSE_COL,
                STANDARD_HUMAN_LABEL_COL,
                STANDARD_DATA_TYPE_COL,  # Optional column for data type
            }
            
            unexpected_columns = columns - allowed_columns
            assert not unexpected_columns, (
                f"CSV {csv_file.name} has unexpected columns: {unexpected_columns}. "
                f"Allowed columns: {allowed_columns}"
            )


class TestCSVVersionMetadata:
    """Test that all CSV files have version metadata."""

    @pytest.fixture(scope="class")
    def all_csv_files(self) -> List[Path]:
        """Get all CSV files from all scorer eval directories."""
        files = []
        files.extend(Path(SCORER_EVALS_OBJECTIVE_PATH).glob("*.csv"))
        files.extend(Path(SCORER_EVALS_HARM_PATH).glob("*.csv"))
        files.extend(Path(SCORER_EVALS_REFUSAL_SCORER_PATH).glob("*.csv"))
        return files

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_OBJECTIVE_PATH).glob("*.csv"))
        + list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv"))
        + list(Path(SCORER_EVALS_REFUSAL_SCORER_PATH).glob("*.csv")),
        ids=lambda p: f"{p.parent.name}/{p.name}",
    )
    def test_csv_has_dataset_version_line(self, csv_file: Path) -> None:
        """
        Test that each CSV has a dataset_version metadata line.
        
        Version line format: # dataset_version=X.Y
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            assert first_line.startswith("#") and "dataset_version=" in first_line, (
                f"CSV {csv_file} is missing dataset_version metadata line. "
                f"First line should contain '# dataset_version=X.Y', but got: '{first_line}'"
            )

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv")),
        ids=lambda p: f"{p.parent.name}/{p.name}",
    )
    def test_harm_csv_has_harm_definition(self, csv_file: Path) -> None:
        """
        Test that each harm CSV has a harm_definition path in its metadata.
        
        Format: # dataset_version=X.Y, harm_definition=path/to/definition.yaml
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            assert "harm_definition=" in first_line, (
                f"Harm CSV {csv_file} is missing harm_definition metadata. "
                f"First line should contain 'harm_definition=path/to/file.yaml', but got: '{first_line}'"
            )

    @pytest.mark.parametrize(
        "csv_file",
        list(Path(SCORER_EVALS_HARM_PATH).glob("*.csv")),
        ids=lambda p: f"{p.parent.name}/{p.name}",
    )
    def test_harm_definition_file_exists_and_is_valid(self, csv_file: Path) -> None:
        """
        Test that the harm_definition file referenced in each harm CSV exists and is valid YAML.
        
        This validates:
        1. The harm_definition path can be parsed from the CSV
        2. The referenced YAML file exists in the harm_definition directory
        3. The YAML file contains valid harm definition structure
        """
        from pyrit.models.harm_definition import HarmDefinition
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Parse harm_definition from the comment line
        harm_definition_path = None
        content = first_line[1:].strip()  # Remove leading #
        for part in content.split(","):
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip() == "harm_definition":
                    harm_definition_path = value.strip()
                    break
        
        assert harm_definition_path is not None, (
            f"Could not parse harm_definition from {csv_file}. First line: '{first_line}'"
        )
        
        # HarmDefinition.from_yaml will raise FileNotFoundError if file doesn't exist,
        # or ValueError if the YAML is invalid
        try:
            harm_def = HarmDefinition.from_yaml(harm_definition_path)
            assert harm_def.version, f"Harm definition {harm_definition_path} is missing version"
            assert harm_def.category, f"Harm definition {harm_definition_path} is missing category"
            assert len(harm_def.scale_descriptions) > 0, (
                f"Harm definition {harm_definition_path} has no scale descriptions"
            )
        except FileNotFoundError as e:
            pytest.fail(f"Harm definition file not found for {csv_file}: {e}")
        except ValueError as e:
            pytest.fail(f"Invalid harm definition YAML for {csv_file}: {e}")

