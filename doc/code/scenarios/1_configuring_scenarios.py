# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pyrit (3.13.5)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Configuring Scenarios
#
# This notebook demonstrates how to use a composite strategies — the `FoundryStrategy` — to test a target with multiple
# attack strategies.
# A "composite strategy"  This class encapsulates a collection of ScenarioStrategy instances along with
# an auto-generated descriptive name, making it easy to represent both single strategies
# and composed multi-strategy attacks.
#
# The `FoundryScenario` provides a comprehensive testing approach that includes:
# - **Converter-based attacks**: Apply various encoding/obfuscation techniques (Base64, Caesar cipher, etc.)
# - **Multi-turn attacks**: Complex conversational attack strategies (Crescendo, RedTeaming)
# - **Strategy composition**: Combine multiple converters together
# - **Difficulty levels**: Organized into EASY, MODERATE, and DIFFICULT categories
#
# Note that this is not the easiest way to run the FoundryScenario (or any scenario). This is meant to show how you can configure all the components.
#
# ## Setup
#
# First, we'll initialize PyRIT and configure the target we want to test.

# %%
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenario import FoundryScenario, FoundryStrategy, ScenarioCompositeStrategy
from pyrit.scenario.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY, initializers=[])  # type: ignore

objective_target = OpenAIChatTarget()
printer = ConsoleScenarioResultPrinter()

# %% [markdown]
# ## Define Seed Groups
#
# By default, `FoundryScenario` selects four random objectives from HarmBench. Here we'll retrieve only one for demonstration.

# %%
from pyrit.datasets import SeedDatasetProvider
from pyrit.models import SeedGroup
from pyrit.scenario import DatasetConfiguration

datasets = await SeedDatasetProvider.fetch_datasets_async(dataset_names=["harmbench"])
seed_groups: list[SeedGroup] = datasets[0].seed_groups # type: ignore
dataset_config = DatasetConfiguration(seed_groups=seed_groups, max_dataset_size=2)

# %% [markdown]
# ## Select Attack Strategies
#
# You can specify individual strategies or compose multiple converters together.
# The scenario supports three types of strategy specifications:
#
# 1. **Simple strategies**: Individual converter or attack strategies (e.g., `FoundryStrategy.Base64`)
# 2. **Aggregate strategies**: Tag-based groups (e.g., `FoundryStrategy.EASY` expands to all easy strategies)
# 3. **Composite strategies**: Multiple converters applied together (e.g., Caesar + CharSwap)

# %%
scenario_strategies = [
    FoundryStrategy.Base64,  # Simple strategy (auto-wrapped internally)
    FoundryStrategy.Binary,  # Simple strategy (auto-wrapped internally)
    ScenarioCompositeStrategy(strategies=[FoundryStrategy.Caesar, FoundryStrategy.CharSwap]),  # Composed strategy
]

# %% [markdown]
# ## Create and Initialize the Scenario
#
# The scenario needs to be initialized before execution. This builds the atomic attacks based on the selected strategies.

# %%
foundry_scenario = FoundryScenario()
await foundry_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    scenario_strategies=scenario_strategies,
    max_concurrency=10,
    dataset_config=dataset_config,
)

print(f"Created scenario: {foundry_scenario.name}")
print(f"Number of atomic attacks: {foundry_scenario.atomic_attack_count}")

# %% [markdown]
# ## Execute the Scenario
#
# Now we'll run the scenario and print the results. The scenario will:
# 1. Execute each atomic attack sequentially
# 2. Apply the attack strategy to all objectives
# 3. Score the results using the configured scorer
# 4. Aggregate all results into a `ScenarioResult`

# %%
foundry_results = await foundry_scenario.run_async()  # type: ignore
await printer.print_summary_async(foundry_results)  # type: ignore

# %% [markdown]
# ## Alternative: Using Difficulty Levels
#
# Instead of specifying individual strategies, you can use aggregate tags like `EASY`, `MODERATE`, or `DIFFICULT` to test multiple strategies at once.

# %%
# Example: Test all EASY strategies
# easy_scenario = FoundryScenario(
#     objective_target=objective_target,
#     scenario_strategies=[FoundryStrategy.EASY],  # Expands to all easy strategies
#     objectives=objectives,
# )
# await easy_scenario.initialize_async()
# easy_results = await easy_scenario.run_async()
# await printer.print_summary_async(easy_results)

# %% [markdown]
# ## Scenario Resiliency
#
# The `FoundryScenario` supports automatic resume and retry mechanisms:
#
# - **Automatic Resume**: If execution is interrupted, re-running the scenario will continue from where it left off
# - **Retry Mechanism**: Set `max_retries` to automatically retry on transient failures
# - **Progress Tracking**: The scenario tracks completed objectives in memory
#
# For more details on resiliency features, see the [resiliency documentation](../setup/2_resiliency.ipynb).
