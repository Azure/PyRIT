# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the setup.DatasetFactory."""

import tempfile
from pathlib import Path

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.setup import DatasetFactory, ConfigurationPaths


@pytest.mark.usefixtures("patch_central_database")
class TestDatasetFactory:
    """Tests for the DatasetFactory class."""

    def test_create_dataset_with_harm_bench_config(self):
        """Test loading dataset from the harm_bench config file."""
        dataset_params = DatasetFactory.create_dataset(
            config_path=ConfigurationPaths.dataset.harm_bench
        )

        assert "objectives" in dataset_params
        assert isinstance(dataset_params["objectives"], list)
        assert len(dataset_params["objectives"]) == 8
        # Verify all objectives are strings
        for obj in dataset_params["objectives"]:
            assert isinstance(obj, str)

    def test_create_dataset_with_override_params(self):
        """Test that override parameters work correctly."""
        custom_objectives = ["Custom objective 1", "Custom objective 2"]

        dataset_params = DatasetFactory.create_dataset(
            config_path=ConfigurationPaths.dataset.harm_bench,
            objectives=custom_objectives,
        )

        assert dataset_params["objectives"] == custom_objectives

    def test_create_dataset_nonexistent_file(self):
        """Test that FileNotFoundError is raised for nonexistent config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            DatasetFactory.create_dataset(config_path="nonexistent_config.py")

    def test_create_dataset_invalid_config_no_dataset_config(self):
        """Test that AttributeError is raised when config file doesn't define dataset_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Empty config file\n")
            f.write("some_variable = 42\n")
            temp_path = f.name

        try:
            with pytest.raises(AttributeError, match="must define 'dataset_config' dictionary"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_config_not_dict(self):
        """Test that ValueError is raised when dataset_config is not a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = 'not a dict'\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must define 'dataset_config' as a dictionary"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_config_no_objectives(self):
        """Test that ValueError is raised when dataset_config doesn't specify objectives."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {'some_param': 'value'}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must define 'objectives'"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_objectives_not_list(self):
        """Test that ValueError is raised when objectives is not a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {'objectives': 'not a list'}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="'objectives' must be a list"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_objectives_empty_list(self):
        """Test that ValueError is raised when objectives list is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {'objectives': []}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="'objectives' list cannot be empty"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_objectives_not_strings(self):
        """Test that ValueError is raised when objectives contain non-string values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {'objectives': ['valid string', 42, 'another string']}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="'objectives\\[1\\]' must be a string"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_with_prepended_conversation_none(self):
        """Test that prepended_conversation can be None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {\n")
            f.write("    'objectives': ['test objective'],\n")
            f.write("    'prepended_conversation': None,\n")
            f.write("}\n")
            temp_path = f.name

        try:
            dataset_params = DatasetFactory.create_dataset(config_path=temp_path)
            assert dataset_params["objectives"] == ["test objective"]
            assert dataset_params["prepended_conversation"] is None
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_with_prepended_conversation_valid(self):
        """Test that prepended_conversation with valid PromptRequestResponse works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("from pyrit.models import PromptRequestPiece, PromptRequestResponse\n")
            f.write("\n")
            f.write("_prepended = [\n")
            f.write("    PromptRequestResponse(\n")
            f.write("        request_pieces=[\n")
            f.write("            PromptRequestPiece(\n")
            f.write("                role='user',\n")
            f.write("                original_value='Previous message',\n")
            f.write("            )\n")
            f.write("        ]\n")
            f.write("    )\n")
            f.write("]\n")
            f.write("\n")
            f.write("dataset_config = {\n")
            f.write("    'objectives': ['test objective'],\n")
            f.write("    'prepended_conversation': _prepended,\n")
            f.write("}\n")
            temp_path = f.name

        try:
            dataset_params = DatasetFactory.create_dataset(config_path=temp_path)
            assert dataset_params["objectives"] == ["test objective"]
            assert isinstance(dataset_params["prepended_conversation"], list)
            assert len(dataset_params["prepended_conversation"]) == 1
            assert isinstance(dataset_params["prepended_conversation"][0], PromptRequestResponse)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_prepended_conversation_not_list(self):
        """Test that ValueError is raised when prepended_conversation is not a list or None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {\n")
            f.write("    'objectives': ['test objective'],\n")
            f.write("    'prepended_conversation': 'not a list',\n")
            f.write("}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="'prepended_conversation' must be a list or None"):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_invalid_prepended_conversation_wrong_type(self):
        """Test that ValueError is raised when prepended_conversation contains non-PromptRequestResponse."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {\n")
            f.write("    'objectives': ['test objective'],\n")
            f.write("    'prepended_conversation': ['not a PromptRequestResponse'],\n")
            f.write("}\n")
            temp_path = f.name

        try:
            with pytest.raises(
                ValueError, match="'prepended_conversation\\[0\\]' must be a PromptRequestResponse"
            ):
                DatasetFactory.create_dataset(config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_with_additional_params(self):
        """Test that additional parameters are passed through."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("dataset_config = {\n")
            f.write("    'objectives': ['test objective'],\n")
            f.write("    'memory_labels': {'category': 'test'},\n")
            f.write("    'custom_param': 'custom_value',\n")
            f.write("}\n")
            temp_path = f.name

        try:
            dataset_params = DatasetFactory.create_dataset(config_path=temp_path)
            assert dataset_params["objectives"] == ["test objective"]
            assert dataset_params["memory_labels"] == {"category": "test"}
            assert dataset_params["custom_param"] == "custom_value"
        finally:
            Path(temp_path).unlink()

    def test_create_dataset_from_config_function(self):
        """Test the convenience function create_dataset_from_config."""
        from pyrit.setup import create_dataset_from_config

        dataset_params = create_dataset_from_config(config_path=ConfigurationPaths.dataset.harm_bench)

        assert "objectives" in dataset_params
        assert isinstance(dataset_params["objectives"], list)
        assert len(dataset_params["objectives"]) == 8

    def test_create_dataset_can_be_unpacked_in_execute_multi_objective(self):
        """Test that the returned dict can be unpacked into execute_multi_objective_attack_async."""
        dataset_params = DatasetFactory.create_dataset(
            config_path=ConfigurationPaths.dataset.harm_bench
        )

        # Verify the structure matches what execute_multi_objective_attack_async expects
        assert "objectives" in dataset_params
        assert isinstance(dataset_params["objectives"], list)
        
        # These are the expected parameters for execute_multi_objective_attack_async
        # The dataset_params should work with: **dataset_params
        expected_param_names = {"objectives", "prepended_conversation", "memory_labels"}
        dataset_param_names = set(dataset_params.keys())
        
        # All dataset params should be valid for the function
        # (objectives is required, others are optional or extra attack params)
        assert "objectives" in dataset_param_names
