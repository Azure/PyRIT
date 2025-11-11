# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # CyberScenario

# %% [markdown] vscode={"languageId": "plaintext"}
# The CyberScenario is a Scenario that focuses on cyberharms.

# %% [markdown]
# ## Use Case

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenarios import CyberScenario, CyberStrategy
from pyrit.scenarios.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(
    memory_db_type=IN_MEMORY,
)

objective_target = OpenAIChatTarget(model_name="gpt4o")
printer = ConsoleScenarioResultPrinter()

# Create a scenario from the pre-configured Foundry scenario
cyber_scenario = CyberScenario(
    objective_target=objective_target, max_concurrency=10, scenario_strategies=[CyberStrategy.MultiTurn]
)
await cyber_scenario.initialize_async()  # type: ignore

print(f"Created scenario: {cyber_scenario.name}")

# Execute the entire scenario
cyber_results = await cyber_scenario.run_async()  # type: ignore
await printer.print_summary_async(cyber_results)  # type: ignore
