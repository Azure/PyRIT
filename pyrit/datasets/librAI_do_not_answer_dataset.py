# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_librAI_do_not_answer_dataset() -> SeedPromptDataset:
    """
    Fetch the LibrAI 'Do Not Answer' dataset and return it as a SeedPromptDataset.

    Relevant Columns:
    - "risk_area"
    - "types_of_harm"
    - "specific_harms"
    - "question"

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note: For more info
    - Paper       - https://arxiv.org/abs/2308.13387
    - Github      - https://github.com/libr-ai/do-not-answer
    - HF Dataset  - https://huggingface.co/datasets/LibrAI/do-not-answer
    """
    # Load dataset from Hugging Face
    data = load_dataset("LibrAI/do-not-answer", split="train")

    seed_prompts = [
        SeedPrompt(
            value=entry["question"],
            data_type="text",
            name="",
            dataset_name="LibrAI/Do-Not-Answer",
            harm_categories=[entry["risk_area"], entry["types_of_harm"], entry["specific_harms"]],
            description=(
                f"This is a prompt from the 'Do Not Answer' dataset under the risk area: {entry['risk_area']}, "
                f"harm type: {entry['types_of_harm']}, and specific harm: {entry['specific_harms']}."
            ),
            source="https://huggingface.co/datasets/LibrAI/do-not-answer",
        )
        for entry in data
    ]

    # Create a SeedPromptDataset from the list of SeedPrompt instances
    return SeedPromptDataset(prompts=seed_prompts)
