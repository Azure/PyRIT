# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cybersecurity Harms Scenario
# The `CyberScenario` class allows you to test a model's willingness to generate malware.

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

scenario_strategies = [CyberStrategy.FAST]


# Create a scenario from the pre-configured Cyber scenario
cyber_scenario = CyberScenario()

await cyber_scenario.initialize_async(objective_target=objective_target)  # type: ignore

print(f"Created scenario: {cyber_scenario.name}")

# Execute the entire scenario
cyber_results = await cyber_scenario.run_async()  # type: ignore
await printer.print_summary_async(cyber_results)  # type: ignore

# %%
