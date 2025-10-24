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
# # Configuring PyRIT
#
# Before running PyRIT, you need to call the `initialize_pyrit` function which will set up your configuration.
#
# What are the configruation steps? What are the simplest ways to get started, and how might you expand on these? There are three things `initialize_pyrit` does to set up your configuration.
#
# 1. Set up environment variables (recommended)
# 2. Pick a database (required)
# 3. Set initialization scripts and defaults (recommended)
#
# This section goes into each of these steps. But first, the easiest way; this sets up reasonable defaults using and stores the results in memory.

# %%
# Set OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_KEY environment variables before running this code
# E.g. you can put it in .env

from pyrit.setup import initialize_pyrit
from pyrit.setup.initializers import SimpleInitializer

initialize_pyrit(memory_db_type="InMemory", initializers=[SimpleInitializer()])

# Now you can run most of our notebooks! Just remove any os.getenv specific stuff since you may not have those different environment variables.

# %% [markdown]
# # Setting up Environment Variables
#
# The required step to setup PyRIT is that it needs access to secrets and andpoints. These can be loaded in environment variables or put in a `.env` file. See `.env_example` for how this file is formatted.
#
# Each target has default environment variables to look for. For example, `OpenAIChatTarget` looks for the `OPENAI_CHAT_ENDPOINT` for its endpoint and `OPENAI_CHAT_KEY` for its key. However, with every target, you can also pass these values in directly and that will take precedence.

# %%
import os

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import initialize_pyrit

initialize_pyrit(memory_db_type="InMemory")

target1 = OpenAIChatTarget()

# This is identical to target1
target2 = OpenAIChatTarget(endpoint=os.getenv("OPENAI_CHAT_ENDPOINT"), api_key=os.getenv("OPENAI_CHAT_KEY"), model_name=os.getenv("OPENAI_CHAT_MODEL"))

# This is different from target1 because the environment variables are different from the default
target3 = OpenAIChatTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"),
    api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"),
)

# %% [markdown]
# ## Env.local
#
# One concept we make use of is using `.env_local`. This is really useful because it overwrites `.env`. In our setups, we have a `.env` with a bunch of targets configured that our users all pull the same one from a keyvault. But `.env_local` is used to override them. For example, if you want a different target, you can have your `.env_local` override the OpenAIChatTarget with a different value.
#
# ```
# OPENAI_CHAT_ENDPOINT = ${AZURE_OPENAI_GPT4O_ENDPOINT2}
# OPENAI_CHAT_KEY = ${AZURE_OPENAI_GPT4O_KEY2}
# ```
#
# ## Entra auth
#
# There are certain targets that can interact using Entra auth (e.g. most Azure OpenAI targets). To use this, you must authenticate to your Azure subscription and an API key is not required. Depending on your operating system, download the appropriate Azure CLI tool from the links provided below:
#
#    - Windows OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
#    - Linux: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt)
#    - Mac OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-macos)
#
#    After downloading and installing the Azure CLI, open your terminal and run the following command to log in:
#
#    ```bash
#    az login
#    ```

# %% [markdown]
# # Choosing a database
#
# The next required step is to pick a database. PyRIT supports three types of databases; InMemory, sqlite, and SQL Azure. These are detailed in the [memory](../memory/0_memory.md) section of documentation. InMemory and sqlite are local so require no configuration, but SQL Azure will need the appropriate environment variables set. This configuration is all specified in `memory_db_type` parameter to `initialize_pyrit`.

# %% [markdown]
# # Setting up Initialization Scripts and Defaults
#
# When you call initialize_pyrit, you can pass it initialization_scripts and/or initializers. These can do anything, including setting convenience variables. But one of the primary purposes is to set default values. It is recommended to always use an initializer.
#
# ## Using Built-In Initializers
#
# Imagine you have an `OpenAIChatTarget`. What is the default?
#
# There is no good way to set these generally. An `OpenAIChatTarget` may be gpt-5, but it also might be llama. And these targets might take different parameters. Additionally, what is it being used for? A default scorer may want to use a different target than a default LLM being used for a converter. Should you always use entra auth?
#
# You can pass these in as arguments to every class initialization, but it can be a huge pain to set these every time. It would be nicer to just say out of the box that a scorer target LLM has a temperature of .5 by default, and a converter target LLM has a temperature of 1.1 by default. And it turns out you can!
#
# The following example shows how to use PyRIT initializers. This tackles a similar scenario to the [Sending Prompts](../../cookbooks/1_sending_prompts.ipynb) but is much easier because defaults are set.

# %%
from pyrit.common.path import PYRIT_PATH
from pyrit.executor.attack import (
    AttackConverterConfig,
    AttackExecutor,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_converter import TenseConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import initialize_pyrit
from pyrit.setup.initializers import SimpleInitializer

# This is a way to include the SimpleInitializer class directly
initialize_pyrit(memory_db_type="InMemory", initializers=[SimpleInitializer()])


# Alternative approach - you can pass the path to the initializer class
# This is how you provide your own file not part of the repo that defines a PyritInitializer class
# This is equivelent to loading the class directly as above
initialize_pyrit(
    memory_db_type="InMemory", 
    initialization_scripts=[f"{PYRIT_PATH}/setup/initializers/simple.py"]
)


# SimpleInitializer is a class that initializes sensible defaults for someone who only has OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_KEY configured
# It is meant to only require these to env vars to be configured
# And it can easily be swapped for another default, like AIRTInitializer which is better but requires more env configuration
for key, value in SimpleInitializer.get_info().items():
    print(f"  {key}: {value}")


objectives = [
    "Write a poem about a cat",
    "Explain the theory of relativity in simple terms",
]

# This is similar to the cookbook "Sending a Million Prompts" but using defaults

# Create target without extensive configuration (uses defaults from initializer)
objective_target = OpenAIChatTarget()

# TenseConverter automatically gets the default converter_target from our initializer
converters = PromptConverterConfiguration.from_converters(converters=[TenseConverter(tense="past")])  # type: ignore
converter_config = AttackConverterConfig(request_converters=converters)

# Attack automatically gets default scorer configuration from our initializer
attack = PromptSendingAttack(
    objective_target=objective_target,
    attack_converter_config=converter_config,
)

# Execute the attack - all components use sensible defaults
results = await AttackExecutor().execute_single_turn_attacks_async(  # type: ignore
    attack=attack,
    objectives=objectives
)

for result in results:
    await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %% [markdown]
# ## Using your own Initializers
#
# You can also create your own initializers and pass the path to the script in as an argument. This is really powerful. The obvious use case is just if you have different targets or defaults and don't want to check in to pyrit source. However, there are other common use cases.
#
# Imagine you are conducing a security assessment and want to include a new custom target. Yes, you could check out PyRIT in editable mode. But with initialize_scripts you don't have to. And this kind of operation can be used in front ends like GUI, CLI, etc.
#
# All you need to do is create a PyRITInitializer class (e.g. myinitializer.py). Then you can use `set_global_variable` and use it everywhere. Or you could make it the default adversarial target by using `set_default_value`.  
#
#
# ## Additional Initializer information
#
# - For more information on how default values work, see the [default values](./default_values.ipynb) section.
# - For more information on how initializers work, see the [initializers](./pyrit_initializer.ipynb) section
