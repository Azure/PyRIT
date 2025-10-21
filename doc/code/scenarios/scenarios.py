# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: pyrit-dev2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scenarios
#
# A `Scenario` is a higher-level construct that groups multiple Attack Configurations together. This allows you to execute a comprehensive testing campaign with multiple attack methods sequentially. Scenarios are meant to be configured and written to test for specific workflows. As such, it is okay to hard code some values.
#
# ## What is a Scenario?
#
# A `Scenario` represents a comprehensive testing campaign composed of multiple atomic attack tests. It orchestrates the execution of multiple `AttackRun`instances sequentially and aggregates the results into a single `ScenarioResult`.
#
# ### Key Components
#
# - **Scenario**: The top-level orchestrator that groups and executes multiple attack runs
# - **AttackRun**: An atomic test unit combining an attack strategy, objectives, and execution parameters
# - **ScenarioResult**: Contains the aggregated results from all attack runs and scenario metadata
#
# ## Use Cases
#
# Some examples of scenarios you might create:
#
# - **VibeCheckScenario**: Randomly selects a few prompts from HarmBench to quickly assess model behavior
# - **QuickViolence**: Checks how resilient a model is to violent objectives using multiple attack techniques
# - **ComprehensiveFoundry**: Tests a target with all available attack converters and strategies.
# - **CustomCompliance**: Tests against specific compliance requirements with curated datasets and attacks
#
# These Scenarios can be updated and added to as you refine what you are testing for.
#
# ## How It Works
#
# Each `Scenario` contains a collection of `AttackRun` objects. When executed:
#
# 1. Each `AttackRun` is executed sequentially
# 2. Every `AttackRun` tests its configured attack against all specified objectives and datasets
# 3. Results are aggregated into a single `ScenarioResult` with all attack outcomes
# 4. Optional memory labels help track and categorize the scenario execution
#
# ## Creating Custom Scenarios
#
# To create a custom scenario, extend the `Scenario` base class. See [`FoundryScenario`](../../../pyrit/scenarios/config/foundry_scenario.py) for an example.
#
# ## Using Scenarios
#
# Scenarios will be exposed for simple runs (e.g. the cli). Below is an example of how to execute them in code.
#

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.executor.attack import ConsoleAttackResultPrinter
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenarios import FoundryAttackStrategy, FoundryScenario

initialize_pyrit(
    memory_db_type=IN_MEMORY,
)

objective_target = OpenAIChatTarget()
printer = ConsoleAttackResultPrinter()

# Create a scenario from the pre-configured Foundry scenario
foundry_scenario = FoundryScenario(objective_target=objective_target, attack_strategies={FoundryAttackStrategy.EASY})

print(f"Created scenario: {foundry_scenario.name}")
print(f"Number of attack runs: {foundry_scenario.attack_run_count}")

# Execute the entire scenario
results = await foundry_scenario.run_async(max_concurrency=5)  # type: ignore

print(f"\nScenario completed with {len(results.attack_results)} total results")
print(f"Success rate: {results.objective_achieved_rate}%\n")

# Print summary for each result
for result in results.attack_results:
    await printer.print_summary_async(result=result)  # type: ignore
