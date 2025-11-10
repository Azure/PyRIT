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
# This notebook demonstrates the usage of the RapidResponseHarmScenario class to test model behavior with respect to various harm categories.

# %% [markdown]
# ## Initialization

# %% [markdown]
# ### Import Required Libraries and Initialize PyRIT
#
#

# %%
from pyrit.memory import CentralMemory
from pyrit.setup.initialization import IN_MEMORY, initialize_pyrit

# Initialize PyRIT with SQLite storage
initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()


# %% [markdown]
# ### Loading the data into memory
#
# Before running the scenario, we need to ensure that the relevant datasets are loaded into memory. In the following cells, we will be testing datasets for hate speech, violence, and harassment strategies.
# %%
from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedDataset

# Import seed prompts
for harm in ["hate", "violence", "harassment"]:
    seed_prompts = SeedDataset.from_yaml_file(Path(DATASETS_PATH) / "seed_prompts" / "harms" / f"{harm}.prompt")
    await memory.add_seeds_to_memory_async(prompts=[*seed_prompts.prompts, *seed_prompts.objectives], added_by="test")  # type: ignore

# %% [markdown]
# ### Running Multiple Harm Strategies
#
# Now we run the strategies using the datasets we defined above!

# %%
import os

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenarios.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.scenarios.scenarios.ai_rt.rapid_response_harm_scenario import (
    RapidResponseHarmScenario,
    RapidResponseHarmStrategy,
)

printer = ConsoleScenarioResultPrinter()

# Create RapidResponseHarmScenario instance for hate speech testing
rapid_response_harm_scenario = RapidResponseHarmScenario(
    objective_target=OpenAIChatTarget(
        endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    ),
    scenario_strategies=[
        RapidResponseHarmStrategy.Hate,
        RapidResponseHarmStrategy.Violence,
        RapidResponseHarmStrategy.Harassment,
    ],
    seed_dataset_name="rapid_response_harm",
)

# Run hate speech tests
print(f"Created scenario: {rapid_response_harm_scenario.name}")
await rapid_response_harm_scenario.initialize_async()  # type: ignore

# Execute the entire scenario
rapid_response_harm_results = await rapid_response_harm_scenario.run_async()  # type: ignore
await printer.print_summary_async(rapid_response_harm_results)  # type: ignore
