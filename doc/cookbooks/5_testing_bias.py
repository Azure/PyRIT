# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pyrit
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing for Fairness and Bias with FairnessBiasBenchmark
#
# In this notebook we show how PyRIT's `FairnessBiasBenchmark` can be used to help test for fairness and bias in an experimental way.
#
# The benchmark provides a clean interface to:
# 1) Generate fictional biographies for given professions/roles, etc
# 2) Score responses from LLMs on configurable factors (pronouns, bias categories, etc.)
# 3) Track and analyze results across multiple experiments
#
# What is important to note is that you can customize the scorer and categories you are testing for! The benchmark requires you to provide a scorer upfront, making the evaluation criteria explicit.

# %%
import pathlib
from collections import Counter

import pandas as pd

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import ConsoleAttackResultPrinter
from pyrit.executor.benchmark import FairnessBiasBenchmark
from pyrit.memory import CentralMemory
from pyrit.models import SeedPromptDataset
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskCategoryScorer

initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()

# %% [markdown]
# ## 1) Set up target, scorer, and benchmark dataset
#
# We are using a list of professions as an independent variable and a scorer for pronouns associated with the response. You can customize to instead look for other values of interest!

# %%
# Set up the target LLM
objective_target = OpenAIChatTarget()

# Create the pronoun scorer (required for the benchmark)
pronoun_scorer = SelfAskCategoryScorer(
    content_classifier_path=pathlib.Path(DATASETS_PATH) / "score" / "categories" / "pronoun_category_score.yaml",
    chat_target=objective_target,
)

# Create the benchmark with the required scorer
benchmark = FairnessBiasBenchmark(objective_target=objective_target, scorer=pronoun_scorer)

print("Benchmark created with pronoun scorer")

# %% [markdown]
# ## 2) Load professions dataset
#
# We'll use the same professions dataset as in the original example to maintain consistency.

# %%
# Load the professions dataset
seed_prompts = SeedPromptDataset.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "lexicons" / "fairness" / "gendered_professions.yaml"
)

await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts.prompts, added_by="fairness_bias_demo")  # type: ignore

# Extract profession names
professions = []
prompt_groups = memory.get_seed_prompt_groups(dataset_name="2025_08_airt_fairness_gendered_professions")
for prompt_group in prompt_groups:
    professions.append(prompt_group.prompts[0].value)

professions

# %% [markdown]
# ## 3) Run bias testing experiments with the benchmark
#
# Now we'll use the benchmark to run experiments across different professions and story types. The benchmark handles the prompt generation, execution, and scoring automatically.

# %%
# Configuration
num_experiments = 2  # Number of experiments per profession per story type
story_types = ["poem with two stanzas", "epic tale in four paragraphs"]

# Storage for results
all_results: dict[str, dict[str, list]] = {}
summary_data: list[dict[str, str | int]] = []

for story_type in story_types:
    all_results[story_type] = {}

    for profession in professions[:3]:  # For demo purposes we just show first 3 professions

        # Run the benchmark for this profession and story type
        result = await benchmark.execute_async(  # type: ignore
            subject=profession, story_type=story_type, num_experiments=num_experiments
        )  # type: ignore
        await ConsoleAttackResultPrinter().print_conversation_async(result)  # type: ignore

        # Get the context and results
        context = benchmark.get_last_context()
        if context:
            # Store the detailed results
            all_results[story_type][profession] = context.experiment_results

            # Extract score categories for analysis
            score_categories = [r["score_category"] for r in context.experiment_results]
            score_counts = Counter(score_categories)

            # Add to summary data
            for category, count in score_counts.items():
                proportion = count / len(score_categories)
                summary_data.append(
                    {
                        "Story Type": story_type,
                        "Profession": profession,
                        "Score Category": category,
                        "Count": count,
                        "Proportion": f"{proportion:.2%}",
                        "Total Experiments": len(score_categories),
                    }
                )

# %% [markdown]
# ## 4) Analyze and visualize results
#
# Let's examine the results using the benchmark's built-in summary functionality and create comprehensive visualizations.

# %%
# Create summary DataFrames for each story type
summary_dfs = {}

for story_type in story_types:
    print(f"Results for '{story_type}':")

    # Filter summary data for this story type
    story_data = [row for row in summary_data if row["Story Type"] == story_type]

    # Create DataFrame
    df = pd.DataFrame(story_data)

    # Calculate totals
    total_experiments = df["Count"].sum()
    total_row = pd.DataFrame(
        [
            {
                "Story Type": story_type,
                "Profession": "TOTAL",
                "Score Category": "All",
                "Count": total_experiments,
                "Proportion": "100.00%",
                "Total Experiments": total_experiments,
            }
        ]
    )

    # Combine and store
    df_with_total = pd.concat([df, total_row], ignore_index=True)
    summary_dfs[story_type] = df_with_total

    # Display the results
    print(df_with_total[["Profession", "Score Category", "Count", "Proportion"]].to_string(index=False))
