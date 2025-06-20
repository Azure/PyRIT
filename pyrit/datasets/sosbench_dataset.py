# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_sosbench_dataset() -> SeedPromptDataset:
    """
    Fetch SOSBench dataset and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information about the dataset and related materials, visit: \n
        - https://huggingface.co/datasets/SOSBench/SOSBench \n
        - https://arxiv.org/abs/2505.21605 \n
    """
    data = load_dataset("apart/sosbench", "default", data_files="sosbench.tsv")

    seed_prompts = [
        SeedPrompt(
            value=item["Example"],
            data_type="text",
            name="",
            dataset_name="SOSBench",
            harm_categories=[item["subject"]],
            description=(
                "SOSBench is a regulation-grounded, hazard-focused benchmark encompassing "
                "six high-risk scientific domains: chemistry, biology, medicine, pharmacology, "
                "physics, and psychology. The benchmark comprises 3,000 prompts derived from "
                "real-world regulations and laws, systematically expanded via an LLM-assisted "
                "evolutionary pipeline that introduces diverse, realistic misuse scenarios"
                " (e.g., detailed explosive synthesis instructions involving advanced"
                " chemical formulas)."
            ),
            source="https://huggingface.co/datasets/SOSBench/SOSBench",
            authors=[
                "Fengqing Jiang",
                "Fengbo Ma",
                "Zhangchen Xu",
                "Yuetai Li",
                "Bhaskar Ramasubramanian",
                "Luyao Niu",
                "Bo Li",
                "Xianyan Chen",
                "Zhen Xiang",
                "Radha Poovendran",
            ],
        )
        for item in data
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
