# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import Literal, Optional

import pycountry

from pyrit.datasets.dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_seclists_bias_testing_dataset(
    source: str = (
        "https://raw.githubusercontent.com/danielmiessler/SecLists/4e747a4/Ai/LLM_Testing/Bias_Testing/"
        "nationality_geographic_bias.txt"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    random_seed: Optional[int] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    nationality: Optional[str] = None,
    gender: Optional[str] = None,
    skin_color: Optional[str] = None,
) -> SeedPromptDataset:
    """
    Fetch SecLists AI LLM Bias Testing examples from a specified source and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the SecLists repository Bias_Testing.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        random_seed (Optional[int]): Seed for random number generation for reproducibility. Defaults to None.
        country (Optional[str]): Specific country to use for the placeholder. Defaults to None.
        region (Optional[str]): Specific region to use for the placeholder. Defaults to None.
        nationality (Optional[str]): Specific nationality to use for the placeholder. Defaults to None.
        gender (Optional[str]): Specific gender to use for the placeholder. Defaults to None.
        skin_color (Optional[str]): Specific skin color to use for the placeholder. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples with placeholders replaced.
    """

    if random_seed is not None:
        random.seed(random_seed)

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Define placeholder values generation functions using pycountry and random.choice
    countries = list(pycountry.countries)
    subdivisions = list(pycountry.subdivisions)

    placeholder_values = {
        "Country": lambda: country if country else random.choice(countries).name,
        "Region": lambda: region if region else random.choice(subdivisions).name,
        "Nationality": lambda: (
            nationality if nationality else random.choice(countries).name
        ),  # Same as Country, the target LLM should be able to understand the context
        "GENDER": lambda: gender if gender else random.choice(["male", "female", "non-binary"]),
        "SKIN_COLOR": lambda: skin_color if skin_color else random.choice(["white", "black", "brown", "yellow"]),
    }

    # Replace placeholders in the examples with generated values
    filled_examples = []
    for example in examples:
        prompt = example["prompt"]
        for placeholder, generator in placeholder_values.items():
            values_used = set()
            while f"[{placeholder}]" in prompt:
                value = generator()
                # Ensure the new value is not the same as the previous one
                while value in values_used:
                    value = generator()
                values_used.add(value)
                prompt = prompt.replace(f"[{placeholder}]", value, 1)

        filled_examples.append(prompt)

    # Create SeedPrompt instances from each example in 'filled_examples'
    seed_prompts = [
        SeedPrompt(
            value=example,
            data_type="text",
            name="SecLists Bias Testing Examples",
            dataset_name="SecLists Bias Testing Examples",
            harm_categories=["bias_testing"],
            description="A dataset of SecLists AI LLM Bias Testing examples with placeholders replaced.",
        )
        for example in filled_examples
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset
