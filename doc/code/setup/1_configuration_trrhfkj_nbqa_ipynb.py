# %%NBQA-CELL-SEP52c935
# Set OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_KEY environment variables before running this code
# E.g. you can put it in .env

from pyrit.setup import initialize_pyrit
from pyrit.setup.initializers import SimpleInitializer

initialize_pyrit(memory_db_type="InMemory", initializers=[SimpleInitializer()])

# Now you can run most of our notebooks! Just remove any os.getenv specific stuff since you may not have those different environment variables.

# %%NBQA-CELL-SEP52c935
import os

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

target1 = OpenAIChatTarget()

# This is identical to target1 because "OPENAI_CHAT_ENDPOINT" are the names of the default environment variables for OpenAIChatTarget 
target2 = OpenAIChatTarget(endpoint=os.getenv("OPENAI_CHAT_ENDPOINT"), api_key=os.getenv("OPENAI_CHAT_KEY"), model_name=os.getenv("OPENAI_CHAT_MODEL"))

# This is (probably) different from target1 because the environment variables are different from the default
target3 = OpenAIChatTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"),
    api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"),
    model_name=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL2")
)

# %%NBQA-CELL-SEP52c935
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


# Alternative approach - you can pass the path to the initializer class.
# This is how you provide your own file not part of the repo that defines a PyRITInitializer class
# This is equivalent to loading the class directly as above
initialize_pyrit(
    memory_db_type="InMemory", 
    initialization_scripts=[f"{PYRIT_PATH}/setup/initializers/simple.py"]
)


# SimpleInitializer is a class that initializes sensible defaults for someone who only has OPENAI_CHAT_ENDPOINT and OPENAI_CHAT_KEY configured
# It is meant to only require these two env vars to be configured
# It can easily be swapped for another PyRITInitializer, like AIRTInitializer which is better but requires more env configuration
# get_info() is a class method that shows how this initializer configures defaults and what global variables it sets
for key, value in SimpleInitializer.get_info().items():
    print(f"{key}: {value}")


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
