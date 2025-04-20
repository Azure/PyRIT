# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_cbt_bench_dataset(config_name: str = "core_fine_seed") -> SeedPromptDataset:
    """
    Fetch CBT-Bench examples for a specific configuration and create a SeedPromptDataset.

    Args:
        config_name (str): The configuration name to load (default is "core_fine_seed").

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information about the dataset and related materials, visit: \n
        - https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench \n
        - Related to Cognitive Behavioral Therapy benchmarking and psychological safety tasks.

    Citation:
        Zhang, M., Yang, X., Zhang, X., Labrum, T., Chiu, J. C., Eack, S. M., Fang, F., Wang, W. Y., & Chen, Z. Z. (2024).
        CBT-Bench: Evaluating Large Language Models on Assisting Cognitive Behavior Therapy. 
        arXiv preprint arXiv:2410.13218.

    Authors:
        Mian Zhang, Xianjun Yang, Xinlu Zhang, Travis Labrum, Jamie C. Chiu, Shaun M. Eack, 
        Fei Fang, William Yang Wang, Zhiyu Zoey Chen
    """
    try:
        # Load the dataset with the specified configuration
        data = load_dataset("Psychotherapy-LLM/CBT-Bench", config_name, split="train")
    except Exception as e:
        raise ValueError(f"Error loading CBT-Bench dataset with config '{config_name}': {e}")

    seed_prompts = [
        SeedPrompt(
            value=item["situation"],  # Use 'situation' as the main prompt text
            data_type="text",
            name=f"CBT-Bench-{item['id']}",
            dataset_name="CBT-Bench",
            harm_categories=item.get("core_belief_fine_grained", []),
            description=(
                "CBT-Bench is a benchmark dataset designed to evaluate the alignment and therapeutic safety of "
                "Large Language Models (LLMs) in the context of Cognitive Behavioral Therapy (CBT)."
            ),
            source="https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench",
            authors=[
                "Mian Zhang",
                "Xianjun Yang",
                "Xinlu Zhang",
                "Travis Labrum",
                "Jamie C. Chiu",
                "Shaun M. Eack",
                "Fei Fang",
                "William Yang Wang",
                "Zhiyu Zoey Chen"
            ],
        )
        for item in data
    ]

    return SeedPromptDataset(prompts=seed_prompts)