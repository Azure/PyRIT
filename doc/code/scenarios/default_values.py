# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Default Values and Initialization Scripts
#
# When you call initialize_pyrit, you can pass it initialization_scripts. These can do anything, including setting convenience variables. But one of the primary purposes is to set default values.
#
# ## Why is this important?
#
# Imagine you have an `OpenAIChatTarget`. What is the default?
#
# It really depends. An `OpenAIChatTarget` may be gpt-5, but it also might be llama. And these targets might take different parameters. Additionally, what is it being used for? A default scorer may want to use a different target than a default LLM being used for a converter.
#
# It can be a pain to set these every time. It would be nicer to just say out of the box that a scorer target LLM has a temperature of 0 by default, and a converter target LLM has a temperature of .7 by default.
#
# ## How Default Values Work
#
# When an initialization script calls `set_default_value`, it registers a default value for a specific class and parameter combination. These defaults are stored in a global registry and are automatically applied when classes are instantiated.
#
# One of the most important things to understand is that **explicitly provided values always override defaults**. Defaults only apply when:
# 1. A parameter is not provided at all, OR
# 2. A parameter is explicitly set to `None`
#
# If you pass a value (even `0`, `False`, or `""`), that value will be used instead of the default.
#
# ## Using `apply_defaults` Decorator
#
# First, it's good to be selective over which classes can use this. It is very powerful but can also make debugging more difficult.
#
# Classes that want to participate in the default value system use the `@apply_defaults` decorator on their `__init__` method:
#
# ```python
# from pyrit.setup.pyrit_default_value import apply_defaults
#
# class MyConverter(PromptConverter):
#     @apply_defaults
#     def __init__(self, *, converter_target: Optional[PromptChatTarget] = None, temperature: Optional[float] = None):
#         self.converter_target = converter_target
#         self.temperature = temperature
# ```
#
# When you create an instance:
#
# ```python
# # Uses defaults for both parameters (if configured)
# converter1 = MyConverter()
#
# # Uses provided value for converter_target, default for temperature
# converter2 = MyConverter(converter_target=my_target)
#
# # Uses provided values for both (defaults ignored)
# converter3 = MyConverter(converter_target=my_target, temperature=0.8)
# ```
#
# Defaults can be set on parent classes and will apply to subclasses (unless `include_subclasses=False` is specified).
#
# ## Example Initialization Script
#
# As an example, here is an exerpt from the initialization script in setup\config\initialization\defaults\converter_initialization.py. When it is run (e.g. as part of `initialize_pyrit`) it sets all converter LLMs to use this target for their `converter_target`.

# %%
"""
This is a good default converter configuration for PyRIT.
"""
import os

from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.initialization import initialize_pyrit
from pyrit.setup.pyrit_default_value import set_default_value

# This is not necessary usually, but we need it here because we're running as a stand alone
initialize_pyrit(memory_db_type="InMemory")

default_converter_target = OpenAIChatTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    temperature=0.5,
)


set_default_value(
    class_type=PromptConverter, parameter_name="converter_target", value=default_converter_target
)

# %% [markdown]
# ## Using Default Values
#
# As mentioned earlier, many types of classes could have better defaults that can be given. 
#
# We need the ability to pass in a `PromptChatTarget` to a scorer. But we don't want to have to pass it every single time.
#
# For our own use within Microsoft, all our configuration is done in `.env` (see [Populating Secrets](../../../setup/populating_secrets.md)). But then using these, we can create default values that make more sense than we can possibly make at a class level. Our own `.env` may make use of all the variables in `.env_example`. Your own initialization scripts may be different.
#
# This is can be run as part of initialize_pyrit by doing the following.

# %%
from pyrit.executor.attack import PromptSendingAttack, AttackConverterConfig, AttackExecutor, ConsoleAttackResultPrinter
from pyrit.prompt_converter import TenseConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration
from pyrit.setup import initialize_pyrit, IN_MEMORY, InitializationPaths

# list_all_paths() loads all initialization scripts in setup/config
# This sets reasonable defaults if .env is configured similar to .env_example
initialize_pyrit(memory_db_type=IN_MEMORY, initialization_scripts=InitializationPaths.list_all_paths())  # type: ignore

objectives = [
    "Write a poem about a cat",
    "Explain the theory of relativity in simple terms",
]

# this is similar to the cookbook "Sending a Million Prompts" but using defaults

# No extra default here.
objective_target = OpenAIChatTarget()


# TenseConverter uses a default PromptChatTarget that is already configured in setup/config/converter_initialization.py
converters = PromptConverterConfiguration.from_converters(converters=[TenseConverter(tense="past")])  # type: ignore default values applied
converter_config = AttackConverterConfig(request_converters=converters)


# prompt sending attack uses a default scorer_config  from setup/config/scorer_initialization.py
attack = PromptSendingAttack(
    objective_target=objective_target,
    attack_converter_config=converter_config,
)

results = await AttackExecutor().execute_single_turn_attacks_async(  # type: ignore
    attack=attack,
    objectives=objectives
)

for result in results:
    await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
