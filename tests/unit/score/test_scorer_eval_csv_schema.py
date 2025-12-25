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
            if not first_line.startswith("# version="):
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
            if not first_line.startswith("# version="):
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
            if not first_line.startswith("# version="):
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
            if not first_line.startswith("# version="):
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
            if not first_line.startswith("# version="):
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
            if not first_line.startswith("# version="):
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
    def test_csv_has_version_line(self, csv_file: Path) -> None:
        """
        Test that each CSV has a version metadata line.
        
        Version line format: # version=X.Y
        """
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            assert first_line.startswith("# version="), (
                f"CSV {csv_file} is missing version metadata line. "
                f"First line should be '# version=X.Y', but got: '{first_line}'"
            )
