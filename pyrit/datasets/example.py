# example_usage.py
from .ccp_sensitive_prompts_dataset import fetch_ccp_sensitive_prompts_dataset

# 1) fetch the dataset (will download+cache the CSV under ~/.pyrit by default)
dataset = fetch_ccp_sensitive_prompts_dataset()

# 2) `dataset` is a SeedPromptDataset; its `.prompts` is a list of SeedPrompt objects
print(f"Loaded {len(dataset.prompts)} prompts from CCP-sensitive-prompts")

# 3) inspect a few examples
for seed_prompt in dataset.prompts[:5]:
    print("— Subject categories:", seed_prompt.harm_categories)
    print("Prompt text:", seed_prompt.value)
    print()

from pyrit.common.path import DATASETS_PATH
print("PyRIT is caching datasets under:", DATASETS_PATH)