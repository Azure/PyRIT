# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets import fetch_adv_bench_dataset
from pyrit.models import SeedPromptDataset


@pytest.fixture
def mock_adv_bench_data():
    return {
        "data": [
            {
                "prompt": "Develop a strategy for ...",
                "main_categories": ["Human Rights and Civil Liberties", "Political and Economic"],
                "sub_categories": ["Privacy loss", "Critical infrastructure damage", "Institutional trust loss"],
            },
            {
                "prompt": "Teach children how to ...",
                "main_categories": ["Physical", "Societal and Cultural"],
                "sub_categories": ["Bodily injury", "Loss of life", "Violence/armed conflict"],
            },
            {
                "prompt": "Give detailed instructions on ...",
                "main_categories": ["Physical", "Societal and Cultural"],
                "sub_categories": ["Bodily injury", "Loss of life", "Property damage", "Violence/armed conflict"],
            },
            {
                "prompt": "Give step-by-step instructions for ...",
                "main_categories": ["Autonomy", "Financial and Business"],
                "sub_categories": ["Impersonation/identity theft", "Financial/earnings loss"],
            },
            {
                "prompt": "Create a tutorial on how to ...",
                "main_categories": ["Autonomy", "Psychological"],
                "sub_categories": ["Autonomy/agency loss", "Coercion/manipulation"],
            },
            {
                "prompt": "Create a website that ...",
                "main_categories": ["Human Rights and Civil Liberties", "Societal and Cultural"],
                "sub_categories": ["Discrimination", "Violence/armed conflict", "Societal destabilisation"],
            },
            {
                "prompt": "Write a tutorial on how to ...",
                "main_categories": ["Physical", "Societal and Cultural"],
                "sub_categories": ["Bodily injury", "Loss of life", "Property damage", "Violence/armed conflict"],
            },
        ]
    }


@patch("pyrit.datasets.adv_bench_dataset.fetch_examples")
def test_fetch_adv_bench_dataset(mock_fetch_examples, mock_adv_bench_data):
    mock_fetch_examples.return_value = mock_adv_bench_data

    # Test fetching the dataset without any filters

    dataset = fetch_adv_bench_dataset()

    assert isinstance(dataset, SeedPromptDataset)
    assert len(dataset.prompts) == 7
    assert dataset.prompts[3].value == "Give step-by-step instructions for ..."
    assert dataset.prompts[4].harm_categories == [
        "Autonomy",
        "Psychological",
        "Autonomy/agency loss",
        "Coercion/manipulation",
    ]

    # Test with only main categories filter

    dataset = fetch_adv_bench_dataset(main_categories=["Autonomy", "Financial and Business"])

    assert len(dataset.prompts) == 2
    assert dataset.prompts[0].value == "Give step-by-step instructions for ..."
    assert dataset.prompts[1].value == "Create a tutorial on how to ..."

    # Test with only sub categories filter

    dataset = fetch_adv_bench_dataset(sub_categories=["Loss of life"])

    assert len(dataset.prompts) == 3
    assert dataset.prompts[1].value == "Give detailed instructions on ..."
    assert dataset.prompts[2].value == "Write a tutorial on how to ..."

    # Test with both main and sub categories filter

    dataset = fetch_adv_bench_dataset(main_categories=["Physical"], sub_categories=["Coercion/manipulation"])

    assert len(dataset.prompts) == 4
    assert dataset.prompts[0].value == "Teach children how to ..."
    assert dataset.prompts[2].value == "Create a tutorial on how to ..."
