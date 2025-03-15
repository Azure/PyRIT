# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets import fetch_pku_safe_rlhf_dataset
from pyrit.models import SeedPromptDataset


@pytest.fixture
def mock_pku_safe_rlhf_data():
    return {
        "train": [
            {
                "prompt": "How do I ...?",
                "is_response_0_safe": True,
                "is_response_1_safe": True,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
            },
            {
                "prompt": "As a manager of a company, I want to ...?",
                "is_response_0_safe": False,
                "is_response_1_safe": False,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": True,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": True,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": True,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": True,
                    "Animal Abuse": False,
                },
            },
            {
                "prompt": "How can I design ...?",
                "is_response_0_safe": False,
                "is_response_1_safe": False,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": True,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": True,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": True,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": True,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": True,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
            },
            {
                "prompt": "If someone doesn't like their job, how can they ...?",
                "is_response_0_safe": True,
                "is_response_1_safe": True,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
            },
            {
                "prompt": "My roommate constantly ...",
                "is_response_0_safe": False,
                "is_response_1_safe": True,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": True,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": True,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": False,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": False,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
            },
            {
                "prompt": "My goal is to steal ...",
                "is_response_0_safe": False,
                "is_response_1_safe": False,
                "response_0_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": False,
                    "Economic Crime": True,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": True,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
                "response_1_harm_category": {
                    "Endangering National Security": False,
                    "Insulting Behavior": False,
                    "Discriminatory Behavior": False,
                    "Endangering Public Health": False,
                    "Copyright Issues": False,
                    "Violence": False,
                    "Drugs": False,
                    "Privacy Violation": True,
                    "Economic Crime": True,
                    "Mental Manipulation": False,
                    "Human Trafficking": False,
                    "Physical Harm": False,
                    "Sexual Content": False,
                    "Cybercrime": True,
                    "Disrupting Public Order": False,
                    "Environmental Damage": False,
                    "Psychological Harm": False,
                    "White-Collar Crime": False,
                    "Animal Abuse": False,
                },
            },
        ]
    }


@patch("pyrit.datasets.pku_safe_rlhf_dataset.load_dataset")
def test_fetch_pku_safe_rlhf_dataset(mock_load_dataset, mock_pku_safe_rlhf_data):
    mock_load_dataset.return_value = mock_pku_safe_rlhf_data

    # Test fetching the dataset with default options

    dataset = fetch_pku_safe_rlhf_dataset()

    assert isinstance(dataset, SeedPromptDataset)
    assert len(dataset.prompts) == 6

    assert dataset.prompts[0].value == "How do I ...?"
    assert dataset.prompts[5].value == "My goal is to steal ..."
    assert dataset.prompts[3].harm_categories == []
    assert dataset.prompts[2].harm_categories == ["Cybercrime", "Privacy Violation", "Psychological Harm"]

    # Test without fetching safe prompts

    dataset = fetch_pku_safe_rlhf_dataset(include_safe_prompts=False)
    assert len(dataset.prompts) == 4
    assert dataset.prompts[3].harm_categories == ["Cybercrime", "Economic Crime", "Privacy Violation"]

    # Test with harm category filters

    dataset = fetch_pku_safe_rlhf_dataset(include_safe_prompts=False, filter_harm_categories=["Economic Crime"])
    assert len(dataset.prompts) == 2
    dataset = fetch_pku_safe_rlhf_dataset(
        include_safe_prompts=False, filter_harm_categories=["White-Collar Crime", "Violence"]
    )
    assert len(dataset.prompts) == 2
