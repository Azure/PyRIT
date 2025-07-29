# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Literal, Optional

from pyrit.datasets.dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from pyrit.models import HarmCategory, SeedPrompt, SeedPromptDataset


def fetch_xstest_dataset(
    source: str = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/a3bb396/xstest_v2_prompts.csv",
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch XSTest examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the exaggerated-safety repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/paul-rottger/exaggerated-safety
    """

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    seed_prompts = []

    for example in examples:
        prompt_text = example["prompt"]
        note = example.get("note", "")

        try:
            harm_category = HarmCategory.parse(note)
        except Exception:
            harm_category = HarmCategory.OTHER

        seed_prompts.append(
            SeedPrompt(
                value=prompt_text,
                data_type="text",
                name="XSTest Examples",
                dataset_name="XSTest Examples",
                harm_categories=[harm_category],
                description="A dataset of XSTest examples containing various categories such as violence, drugs, etc.",
            )
        )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset
