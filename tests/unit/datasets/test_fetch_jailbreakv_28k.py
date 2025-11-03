# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from pyrit.datasets.fetch_jailbreakv_28k_dataset import fetch_jailbreakv_28k_dataset
from pyrit.models import SeedDataset, SeedPrompt


class TestFetchJailbreakv28kDataset:
    """Test suite for the fetch_jailbreakv_28k_dataset function."""

    @pytest.mark.parametrize("text_field", [None, "jailbreak_query"])
    @pytest.mark.parametrize(
        "harm_categories",
        [None, ["Economic Harm"], ["Government Decision"]],
    )
    @pytest.mark.parametrize("min_prompts", [0, 2, 5])
    @patch("pyrit.datasets.fetch_jailbreakv_28k_dataset.pathlib.Path")
    @patch("pyrit.datasets.fetch_jailbreakv_28k_dataset._resolve_image_path")
    @patch("pyrit.datasets.fetch_jailbreakv_28k_dataset.load_dataset")
    def test_fetch_jailbreakv_28k_dataset_success(
        self, mock_load_dataset, mock_resolve_image_path, mock_pathlib, text_field, harm_categories, min_prompts
    ):
        # Mock Path to simulate zip file exists and is already extracted
        mock_zip_path = MagicMock()
        mock_zip_path.exists.return_value = True
        mock_extracted_path = MagicMock()
        mock_extracted_path.exists.return_value = True

        mock_pathlib.Path.return_value.__truediv__.side_effect = [
            mock_zip_path,  # First call: zip_file_path
            mock_extracted_path,  # Second call: zip_extracted_path
            mock_extracted_path,  # Additional calls for image resolution
        ]
        # Mock dataset response
        mock_dataset = {
            "mini_JailBreakV_28K": [
                {
                    "redteam_query": "test query 1",
                    "jailbreak_query": "jailbreak: test query 1",
                    "policy": "Economic Harm",
                    "image_path": "mock_folder/valid",
                },
                {
                    "redteam_query": "test query 2",
                    "jailbreak_query": "jailbreak: test query 2",
                    "policy": "Government Decision",
                    "image_path": "invalid",
                },
                {
                    "redteam_query": "test query 3",
                    "jailbreak_query": "jailbreak: test query 3",
                    "policy": "Fraud",
                    "image_path": "mock_folder/valid",
                },
            ]
        }
        mock_load_dataset.return_value = mock_dataset

        def fake_resolve_image_path(
            *, rel_path: str = "", local_directory: pathlib.Path = pathlib.Path(), **kwargs
        ) -> str:
            return "" if rel_path == "invalid" else f"mock_path/{rel_path}"

        mock_resolve_image_path.side_effect = fake_resolve_image_path

        # Call the function
        # Select context: expect error only for this filter
        expect_error = (
            harm_categories == ["Government Decision"]
            or (min_prompts == 1 and harm_categories == ["Government Decision"])
            or min_prompts == 5
        )
        ctx = pytest.raises(ValueError) if expect_error else nullcontext()

        # Single call
        with ctx:
            result = fetch_jailbreakv_28k_dataset(
                text_field=text_field, harm_categories=harm_categories, min_prompts=min_prompts
            )
        if expect_error:
            return
        # Assertions

        assert isinstance(result, SeedDataset)
        if harm_categories is None:
            assert len(result.prompts) == 4
            assert sum(p.data_type == "text" for p in result.prompts) == 2
            assert sum(p.data_type == "image_path" for p in result.prompts) == 2
        elif harm_categories == ["Economic Harm"]:
            assert len(result.prompts) == 2
            assert sum(p.data_type == "text" for p in result.prompts) == 1
            assert sum(p.data_type == "image_path" for p in result.prompts) == 1
            print(result.prompts)
            assert result.prompts[0].harm_categories == ["economic_harm"]
        assert all(isinstance(prompt, SeedPrompt) for prompt in result.prompts)
        print(result.prompts)
        if text_field == "jailbreak_query":
            for prompt in result.prompts:
                if prompt.data_type == "text":
                    assert "jailbreak" in prompt.value
        else:
            for prompt in result.prompts:
                if prompt.data_type == "text":
                    assert "jailbreak" not in prompt.value
