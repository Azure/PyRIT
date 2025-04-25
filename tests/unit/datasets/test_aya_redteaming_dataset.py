# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets import fetch_aya_redteaming_dataset
from pyrit.models import SeedPromptDataset


@pytest.fixture
def mock_aya_redteaming_data():
    return [
        {
            "prompt": "[Prompt #001] ...",
            "language": "English",
            "harm_category": '["Discrimination & Injustice"]',
            "global_or_local": "global",
        },
        {
            "prompt": "[Prompt #002] ...",
            "language": "English",
            "harm_category": '["Harms of Representation Allocation and Quality of Service", '
            '"Discrimination & Injustice"]',
            "global_or_local": "global",
        },
        {
            "prompt": "[Prompt #003] ...",
            "language": "English",
            "harm_category": '["Non-consensual sexual content"]',
            "global_or_local": "local",
        },
        {
            "prompt": "[Prompt #004] ...",
            "language": "English",
            "harm_category": '["Self-Harm", "Violence, Threats & Incitement"]',
            "global_or_local": "local",
        },
        {
            "prompt": "[Prompt #005] ...",
            "language": "English",
            "harm_category": '["Violence, Threats & Incitement", "Bullying & Harassment", "Profanity"]',
            "global_or_local": "global",
        },
        {
            "prompt": "[Prompt #006] ...",
            "language": "English",
            "harm_category": '["Violence, Threats & Incitement", "Graphic material"]',
            "global_or_local": "global",
        },
        {
            "prompt": "[Prompt #007] ...",
            "language": "English",
            "harm_category": '["Bullying & Harassment"]',
            "global_or_local": "global",
        },
    ]


@patch("pyrit.datasets.aya_redteaming_dataset.fetch_examples")
def test_fetch_aya_redteaming_dataset(mock_fetch_examples, mock_aya_redteaming_data):
    mock_fetch_examples.return_value = mock_aya_redteaming_data

    # Test fetching the dataset without any filters

    dataset = fetch_aya_redteaming_dataset()

    assert isinstance(dataset, SeedPromptDataset)
    assert len(dataset.prompts) == 7

    first_prompt = dataset.prompts[0]
    assert first_prompt.value == "[Prompt #001] ..."
    assert first_prompt.data_type == "text"
    assert first_prompt.name == "Aya Red-teaming Examples"
    assert first_prompt.dataset_name == "Aya Red-teaming Examples"
    assert first_prompt.harm_categories == ["Discrimination & Injustice"]
    assert first_prompt.source == "https://huggingface.co/datasets/CohereForAI/aya_redteaming"

    assert dataset.prompts[1].value == "[Prompt #002] ..."
    assert dataset.prompts[3].harm_categories == ["Self-Harm", "Violence, Threats & Incitement"]

    # Test fetching the dataset with a `harm_categories` filter

    dataset = fetch_aya_redteaming_dataset(harm_categories=["Bullying & Harassment"])

    assert len(dataset.prompts) == 2
    assert dataset.prompts[0].value == "[Prompt #005] ..."

    dataset = fetch_aya_redteaming_dataset(harm_categories=["Discrimination & Injustice", "Graphic material"])

    assert len(dataset.prompts) == 3
    assert dataset.prompts[0].value == "[Prompt #001] ..."
    assert dataset.prompts[2].value == "[Prompt #006] ..."

    # Test fetching the dataset with a `harm_scope` filter

    dataset = fetch_aya_redteaming_dataset(harm_scope="local")

    assert len(dataset.prompts) == 2
    assert dataset.prompts[0].value == "[Prompt #003] ..."

    # Test fetching the dataset with multiple filters

    dataset = fetch_aya_redteaming_dataset(harm_categories=["Violence, Threats & Incitement"], harm_scope="global")

    assert len(dataset.prompts) == 2
    assert dataset.prompts[0].value == "[Prompt #005] ..."

    # Test fetching the dataset with a `language` filter

    dataset = fetch_aya_redteaming_dataset(language="French")

    call_args = mock_fetch_examples.call_args
    assert "aya_fra.jsonl" in call_args.kwargs["source"]
