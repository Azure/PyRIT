# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Literal, Optional

from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_mlcommons_ailuminate_demo_dataset(
    source: str = (
        "https://raw.githubusercontent.com/mlcommons/ailuminate/refs/heads/main/"
        "airr_official_1.0_demo_en_us_prompt_set_release.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch examples from AILuminate v1.0 DEMO Prompt Set and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """
    examples = fetch_examples(source, source_type, cache, data_home)

    seed_prompts = [
        SeedPrompt(
            value=example["prompt_text"],
            data_type="text",
            name="",
            dataset_name="AILuminate v1.0 DEMO Prompt Set",
            harm_categories=[example["hazard"]],
            description=(
                "This dataset contains the DEMO prompt library of the AILuminate 1.0 prompt dataset, created by"
                " MLCommons AI Risk & Reliability working group. It contains 1,200 human-generated prompts that"
                " cover 12 different hazard categories. The dataset is a mix of adversarial and non-adversarial"
                " prompts. This dataset is a 10% subset of the full AILuminate training dataset."
            ),
            source="https://github.com/mlcommons/ailuminate",
        )
        for example in examples
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
