# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import List, Literal, Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets.dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_decoding_trust_stereotypes_dataset(
    source: str = (
        "https://raw.githubusercontent.com/AI-secure/DecodingTrust/main/data/stereotype/dataset/user_prompts.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    stereotype_topics: Optional[List[str]] = None,
    target_groups: Optional[List[str]] = None,
    system_prompt_type: Literal["benign", "untargeted", "targeted"] = "targeted",
) -> SeedPromptDataset:
    """
    Fetch DecodingTrust examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the DecodingTrust repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        stereotype_topics (Optional[List[str]]): List of stereotype topics to filter the examples. Defaults to None.
            The list of all 16 stereotype_topics can be found here:
            https://github.com/AI-secure/DecodingTrust/tree/main/data/stereotype
            Defaults to None, which means all topics are included.
        target_groups (Optional[List[str]]): List of target groups to filter the examples. Defaults to None.
            The list of all 24 target_groups can be found here:
            https://github.com/AI-secure/DecodingTrust/tree/main/data/stereotype
            Defaults to None, which means all target groups are included.
        system_prompt_type (Literal["benign", "untargeted", "targeted"]): The type of system prompt to use.
            Defaults to "targeted".

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/centerforaisafety/HarmBench
    """
    if system_prompt_type not in ["benign", "untargeted", "targeted"]:
        raise ValueError("Invalid system_prompt_type. Expected 'benign', 'untargeted', or 'targeted'}")
    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    if system_prompt_type == "targeted":
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_targeted.yaml"
        )
    elif system_prompt_type == "untargeted":
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_untargeted.yaml"
        )
    else:
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_benign.yaml"
        )

    # Required keys to validate each example
    required_keys = {"stereotype_topic", "target_group", "user_prompt"}

    # Initialize containers for prompts and semantic categories
    prompts = []

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    is_targeted = system_prompt_type == "targeted"
    # Validate each example and extract data
    for example in examples:
        # Check for missing keys in the example
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")
        # Check if the example should be filtered based on the provided topics and target groups
        if (
            stereotype_topics
            and example["stereotype_topic"] not in stereotype_topics
            or (target_groups and example["target_group"] not in target_groups)
        ):
            continue

        prompt = (
            system_prompt.render_template_value(prompt=example["user_prompt"], target_group=example["target_group"])
            if is_targeted
            else system_prompt.render_template_value(prompt=example["user_prompt"])
        )
        prompts.append(prompt)

    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="DecodingTrust Stereotypes Examples",
            dataset_name="DecodingTrust Stereotypes Examples",
            harm_categories=["stereotypes"],
            description="""A dataset of Decoding Trust 'Stereotypes' example prompts across many different topics
            and target groups. These prompts attempt to convince the target to agree with a harmful
            stereotype.""",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
