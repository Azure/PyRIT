# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
from uuid import uuid4

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_red_team_social_bias_dataset() -> SeedPromptDataset:
    """
    Fetch Red Team Social Bias Prompts and create a SeedPromptDataset.

    Relevant Columns:
    - "organization"
    - "categorization"
    - "prompt_type"
    - "prompt_instruction"
    - "prompt"
    - "explanation"
    - "ai_response"

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://svannie678.github.io/svannie678-red_team_repo_social_bias. This dataset
        contains aggregated and unified existing red-teaming prompts designed to identify
        stereotypes, discrimination, hate speech, and other representationharms in text-based
        Large Language Models (LLMs).
        Written by Simone Van Taylor.
    """
    data = load_dataset("svannie678/red_team_repo_social_bias_prompts", "default")

    # Define common metadata that will be used across all seed prompts
    common_metadata = {
        "dataset_name": "Red team social bias prompts",
        "authors": ["Simone van Taylor"],
        "description": """This dataset contains aggregated and unified existing
                red-teaming prompts designed to identify stereotypes,
                discrimination, hate speech, and other representation
                harms in text-based Large Language Models (LLMs).""",
        "source": "https://huggingface.co/datasets/svannie678/red_team_repo_social_bias_prompts",
    }

    # Initialize an empty list to store the seed prompts
    seed_prompts = []

    for item in data["train"]:

        # This dataset contains 3 prompt types: "Single Prompt", "Multi Turn" and "Multi Turn, Single Prompt"
        # We're only checking for "Multi Turn" because the "Multi Turn, Single Prompt"
        # prompt types inside this dataset will be difficult to parse
        prompt_type = item.get("prompt_type")

        if prompt_type is None:
            continue

        # Dictionary of metadata for the current prompt
        prompt_metadata = {
            **common_metadata,
            "harm_categories": (
                [item["categorization"]]
                if not isinstance(item.get("categorization"), list)
                else item.get("categorization", [])
            ),
            "groups": [item.get("organization", "")],
            "metadata": {
                "prompt_type": prompt_type,
            },
        }

        if prompt_type in ["Multi Turn"]:
            try:
                # Safely parse the user prompts, remove the unwanted ones such as "assistant" and "system"
                user_prompts = [
                    turn["body"] for turn in ast.literal_eval(item.get("prompt", "")) if turn["role"].startswith("user")
                ]

                group_id = uuid4()
                for i, user_prompt in enumerate(user_prompts):
                    seed_prompts.append(
                        SeedPrompt(
                            value=user_prompt,
                            data_type="text",
                            prompt_group_id=group_id,
                            sequence=i,
                            **prompt_metadata,  # type: ignore
                        )
                    )
            except Exception as e:
                print(f"Error processing Multi-Turn Prompt: {e}")
        else:
            # Clean up single turn prompts that contain unwanted lines of text
            cleaned_value = item.get("prompt", "").replace("### Response:", "").replace("### Instruction:", "").strip()
            # some entries have contents that trip up jinja2, so we escape them
            escaped_cleaned_value = f"{{% raw %}}{cleaned_value}{{% endraw %}}"
            print(f"Single Turn Prompt: {cleaned_value}")
            print(prompt_metadata)
            seed_prompts.append(
                SeedPrompt(
                    value=escaped_cleaned_value,
                    data_type="text",
                    **prompt_metadata,  # type: ignore
                )
            )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
