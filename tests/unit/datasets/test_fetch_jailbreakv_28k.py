# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets.fetch_jailbreakv_28k_dataset import fetch_jailbreakv_28k_dataset
from pyrit.models import SeedPrompt, SeedPromptDataset


class TestFetchJailbreakv28kDataset:
    """Test suite for the fetch_jailbreakv_28k_dataset function."""

    @pytest.mark.parametrize("text_field", [None, "jailbreak_query"])
    @pytest.mark.parametrize("harm_categories", [None, ["Economic Harm"]])
    @patch("pyrit.datasets.fetch_jailbreakv_28k_dataset.load_dataset")
    def test_fetch_jailbreakv_28k_dataset_success(self, mock_load_dataset, text_field, harm_categories):
        # Mock dataset response
        mock_dataset = {
            "mini_JailBreakV_28K": [
                {
                    "redteam_query": "test query 1",
                    "jailbreak_query": "jailbreak: test query 1",
                    "policy": "Economic Harm",
                },
                {
                    "redteam_query": "test query 2",
                    "jailbreak_query": "jailbreak: test query 2",
                    "policy": "Government Decision",
                },
                {
                    "redteam_query": "test query 3",
                    "jailbreak_query": "jailbreak: test query 3",
                    "policy": "Fraud",
                },
            ]
        }
        mock_load_dataset.return_value = mock_dataset

        # Call the function
        result = fetch_jailbreakv_28k_dataset(text_field=text_field, harm_categories=harm_categories)

        # Assertions
        assert isinstance(result, SeedPromptDataset)
        if harm_categories is None:
            assert len(result.prompts) == 3
        elif harm_categories == ["Economic Harm"]:
            assert len(result.prompts) == 1
            print(result.prompts)
            assert result.prompts[0].harm_categories == ["economic_harm"]
        assert all(isinstance(prompt, SeedPrompt) for prompt in result.prompts)
        print(result.prompts)
        if text_field == "jailbreak_query":
            assert all("jailbreak" in prompt.value for prompt in result.prompts)
        else:
            assert all("jailbreak" not in prompt.value for prompt in result.prompts)
