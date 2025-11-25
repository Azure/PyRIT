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
# # Content Harm Testing
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
# Now we can run the strategies using the datasets we defined above! We can selectively choose which strategies to run. In this example, we'll only run the Hate, Violence, and Harassment strategies.

# %%

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenario.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.scenario.scenarios.airt import (
    ContentHarmScenario,
    ContentHarmStrategy,
)
from pyrit.score.true_false.self_ask_refusal_scorer import SelfAskRefusalScorer
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer

printer = ConsoleScenarioResultPrinter()
objective_target = OpenAIChatTarget()

# Create ContentHarmScenario instance for hate, violence, and harassment testing
content_harm_scenario = ContentHarmScenario(
    objective_scorer=TrueFalseInverterScorer(scorer=SelfAskRefusalScorer(chat_target=OpenAIChatTarget())),
    # Uncomment the following line to use a custom dataset prefix, we're using the default here
    # seed_dataset_prefix=="custom_prefix",
)
await content_harm_scenario.initialize_async(  # type: ignore
    scenario_strategies=[
        ContentHarmStrategy.Hate,
        ContentHarmStrategy.Violence,
        ContentHarmStrategy.Harassment,
    ],
    objective_target=objective_target,
)
print(f"Created scenario: {content_harm_scenario.name}")

# Execute the entire scenario
content_harm_results = await content_harm_scenario.run_async()  # type: ignore
await printer.print_summary_async(content_harm_results)  # type: ignore


# %% [markdown]
# We can then inspect individual results from the different harm categories. Below is an example of printing out the results from the Violence strategy.
# %%


from pyrit.executor.attack.printer.console_printer import ConsoleAttackResultPrinter

for result in content_harm_results.attack_results["violence"]:
    await ConsoleAttackResultPrinter().print_summary_async(result=result)  # type: ignore
