# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import List, Literal, Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_adv_bench_dataset(
    cache: bool = True,
    main_categories: Optional[
        List[
            Literal[
                "Autonomy",
                "Physical",
                "Psychological",
                "Reputational",
                "Financial and Business",
                "Human Rights and Civil Liberties",
                "Societal and Cultural",
                "Political and Economic",
                "Environmental",
            ]
        ]
    ] = None,
    sub_categories: Optional[List[str]] = None,
) -> SeedPromptDataset:
    """
    Retrieve AdvBench examples enhanced with categories from a collaborative and human-centered harms taxonomy.

    This function fetches a dataset extending the original AdvBench Dataset by adding harm types to each prompt.
    Categorization was done using the Claude 3.7 model based on the Collaborative, Human-Centered Taxonomy of AI,
    Algorithmic, and Automation Harms (https://arxiv.org/abs/2407.01294v2). Each entry includes at least one main
    category and one subcategory to enable better filtering and analysis of the dataset.

    Useful link: https://arxiv.org/html/2407.01294v2/x2.png (Overview of the Harms Taxonomy)

    Args:
        cache (bool): Whether to cache the fetched examples. Defaults to True.

        main_categories (Optional[List[str]]): A list of main harm categories to search for in the dataset.
            For descriptions of each category, see the paper: arXiv:2407.01294v2
            Defaults to None, which includes all 9 main categories.

        sub_categories (Optional[List[str]]): A list of harm subcategories to search for in the dataset.
            For the complete list of all subcategories, see the paper: arXiv:2407.01294v2.
            Defaults to None, which includes all subcategories.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench. Based on research in paper:
        https://arxiv.org/abs/2307.15043 written by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
        J. Zico Kolter, Matt Fredrikson.

        The categorization approach was proposed by @paulinek13, who suggested using the Collaborative, Human-Centred
        Taxonomy of AI, Algorithmic, and Automation Harms (arXiv:2407.01294v2) to classify the AdvBench examples and
        used Anthropic's Claude 3.7 Sonnet model to perform the categorization based on the taxonomy's descriptions.
    """
    dataset = fetch_examples(
        source=str(Path(DATASETS_PATH) / "data" / "adv_bench_dataset.json"), source_type="file", cache=cache
    )

    filtered = dataset["data"]  # type: ignore

    if main_categories or sub_categories:
        main_set = set(main_categories or [])
        sub_set = set(sub_categories or [])

        # Include an entry if it matches ANY specified main category OR ANY specified subcategory
        filtered = [
            item
            for item in filtered
            if (main_set and any(cat in main_set for cat in item["main_categories"]))
            or (sub_set and any(cat in sub_set for cat in item["sub_categories"]))
        ]

    seed_prompts = [
        SeedPrompt(
            value=item["prompt"],
            data_type="text",
            name="AdvBench Dataset [Extended]",
            dataset_name="AdvBench Dataset",
            harm_categories=item["main_categories"] + item["sub_categories"],
            description="""AdvBench is a set of 520 harmful behaviors formulated as instructions. This dataset
            has been extended to include harm categories for better filtering and analysis. The adversary's goal
            is instead to find a single attack string that will cause the model to generate any response that
            attempts to comply with the instruction, and to do so over as many harmful behaviors as possible.""",
            source="https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench",
        )
        for item in filtered
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
