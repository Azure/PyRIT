# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_darkbench_dataset() -> SeedPromptDataset:
    """
    Fetch DarkBench examples and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information about the dataset and related materials, visit: \n
        - https://huggingface.co/datasets/apart/darkbench \n
        - https://darkbench.ai/ \n
        - https://openreview.net/forum?id=odjMSBSWRt
    """
    data = load_dataset("apart/darkbench", "default", split="train", data_files="darkbench.tsv")

    seed_prompts = [
        SeedPrompt(
            value=item["Example"],
            data_type="text",
            name="",
            dataset_name="DarkBench",
            harm_categories=[item["Deceptive Pattern"]],
            description=(
                "DarkBench is a comprehensive benchmark designed to detect dark design patterns in large language"
                " models (LLMs). Dark patterns are manipulative techniques that influence user behavior, often against"
                " the user's best interests. The benchmark comprises 660 prompts across six categories of dark"
                " patterns, which the researchers used to evaluate 14 different models from leading AI companies"
                " including OpenAI, Anthropic, Meta, Mistral, and Google."
            ),
            source="https://huggingface.co/datasets/apart/darkbench",
            authors=[
                "Esben Kran",
                "Jord Nguyen",
                "Akash Kundu",
                "Sami Jawhar",
                "Jinsuk Park",
                "Mateusz Maria Jurewicz",
            ],
        )
        for item in data
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
