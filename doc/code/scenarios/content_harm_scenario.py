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
# # Rapid Response Harm Testing
#
# This notebook demonstrates the usage of the ContentHarmScenario class to test model behavior with respect to various harm categories.

# %% [markdown]
# ## Initialization

# %% [markdown]
# ### Import Required Libraries and Initialize PyRIT
#
#

# %%
from pyrit.memory import CentralMemory
from pyrit.setup.initialization import IN_MEMORY, initialize_pyrit

# Initialize PyRIT with IN_MEMORY storage
initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()


# %% [markdown]
# ### Loading the data into memory
#
# Before running the scenario, we need to ensure that the relevant datasets are loaded into memory. We have provided a sample set of harm-related seed prompts and are loading them into memory in the next cell.
# %%
from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedDataset

# Import seed prompts
for harm in ["hate", "violence", "harassment", "leakage", "sexual", "fairness", "misinformation"]:
    seed_prompts = SeedDataset.from_yaml_file(Path(DATASETS_PATH) / "seed_prompts" / "harms" / f"{harm}.prompt")
    await memory.add_seeds_to_memory_async(prompts=[*seed_prompts.prompts, *seed_prompts.objectives], added_by="test")  # type: ignore

# %% [markdown]
# ### Running Multiple Harm Strategies
#
# Now we can run the strategies using the datasets we defined above! In this first example, we'll run all the strategies.

# %%
import os

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenarios.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.scenarios.scenarios.ai_rt.content_harm_scenario import (
    ContentHarmScenario,
    ContentHarmStrategy,
)

printer = ConsoleScenarioResultPrinter()

# Create ContentHarmScenario instance for all harm strategies
content_harm_scenario = ContentHarmScenario(
    objective_target=OpenAIChatTarget(
        endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    ),
    scenario_strategies=[ContentHarmStrategy.ALL],
    # Uncomment the following line to use a custom dataset prefix, we're using the default here
    # seed_dataset_prefix=="custom_prefix",
)

# Run strategies
print(f"Created scenario: {content_harm_scenario.name}")
await content_harm_scenario.initialize_async()  # type: ignore

# Execute the entire scenario
content_harm_results = await content_harm_scenario.run_async()  # type: ignore
await printer.print_summary_async(content_harm_results)  # type: ignore


# %% [markdown]
#
# We can also selectively choose which strategies to run. In this next example, we'll only run the Hate, Violence, and Harassment strategies.

# %%

# Create ContentHarmScenario instance for hate, violence, and harassment testing
content_harm_scenario = ContentHarmScenario(
    objective_target=OpenAIChatTarget(
        endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    ),
    scenario_strategies=[
        ContentHarmStrategy.Hate,
        ContentHarmStrategy.Violence,
        ContentHarmStrategy.Harassment,
    ],
    # Uncomment the following line to use a custom dataset prefix, we're using the default here
    # seed_dataset_prefix=="custom_prefix",
)

# Run strategies
print(f"Created scenario: {content_harm_scenario.name}")
await content_harm_scenario.initialize_async()  # type: ignore

# Execute the entire scenario
content_harm_results = await content_harm_scenario.run_async()  # type: ignore
await printer.print_summary_async(content_harm_results)  # type: ignore


# %% [markdown]
# We can then inspect individual results from the different harm categories. Below is an example of printing out the results from the Violence strategy.
# %%


from pyrit.executor.attack.printer.console_printer import ConsoleAttackResultPrinter


for result in content_harm_results.attack_results['violence']:
    await ConsoleAttackResultPrinter().print_summary_async(result=result) # type: ignore