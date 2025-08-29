# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import tempfile
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory import MemoryExporter, MemoryInterface
from pyrit.models import PromptRequestPiece


def test_export_conversation_by_attack_id_file_created(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    attack1_id = sample_conversations[0].attack_identifier["id"]

    # Default path in export_conversations()
    file_name = f"{attack1_id}.json"
    file_path = Path(DB_DATA_PATH, file_name)

    duckdb_instance.exporter = MemoryExporter()

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get:
        mock_get.return_value = sample_conversations
        duckdb_instance.export_conversations(attack_id=attack1_id, file_path=file_path)

        # Verify file was created
        assert file_path.exists()


def test_export_all_conversations_file_created(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_scores") as mock_get_scores,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [
                MagicMock(
                    original_prompt_id="1234",
                    converted_value="sample piece",
                    to_dict=lambda: {"prompt_request_response_id": "1234", "conversation": ["sample piece"]},
                )
            ]
            mock_get_scores.return_value = [
                MagicMock(
                    prompt_request_response_id="1234",
                    score_value=10,
                    to_dict=lambda: {"prompt_request_response_id": "1234", "score_value": 10},
                )
            ]

            assert file_path.exists()


def test_export_all_conversations_with_scores_correct_data(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()

    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_scores") as mock_get_scores,
            patch.object(duckdb_instance.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [MagicMock(original_prompt_id="1234", converted_value="sample piece")]
            mock_get_scores.return_value = [MagicMock(prompt_request_response_id="1234", score_value=10)]

            duckdb_instance.export_conversations(file_path=file_path)

            pos_arg, named_args = mock_export_data.call_args
            assert str(named_args["file_path"]) == temp_file.file.name
            assert str(named_args["export_type"]) == "json"
            assert pos_arg[0][0].original_prompt_id == "1234"
            assert pos_arg[0][0].converted_value == "sample piece"


def test_export_all_conversations_with_scores_empty_data(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()
    expected_data: Sequence = []
    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_scores") as mock_get_scores,
            patch.object(duckdb_instance.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = []
            mock_get_scores.return_value = []

            duckdb_instance.export_conversations(file_path=file_path)
            mock_export_data.assert_called_once_with(expected_data, file_path=file_path, export_type="json")
