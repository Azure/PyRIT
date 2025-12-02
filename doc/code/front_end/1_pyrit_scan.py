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
# # 1. PyRIT Scan
#
# `pyrit_scan` allows you to run automated security testing and red teaming attacks against AI systems using [scenarios](../scenarios/0_scenarios.ipynb) for strategies and [configuration](../setup/1_configuration.ipynb).
#
# Note in this doc the ! prefaces all commands in the terminal so we can run in a Jupyter Notebook.
#
# ## Quick Start
#
# For help:

# %%
# !pyrit_scan --help

# %% [markdown]
# ### Discovery
#
# List all available scenarios:

# %%
# !pyrit_scan --list-scenarios

# %% [markdown]
# **Tip**: You can also discover user-defined scenarios by providing initialization scripts:
#
# ```shell
# pyrit_scan --list-scenarios --initialization-scripts ./my_custom_initializer.py
# ```
#
# This will load your custom scenario definitions and include them in the list.
#
# ## Initializers
#
# PyRITInitializers are how you can configure the CLI scanner. PyRIT includes several built-in initializers you can use with the `--initializers` flag.
#
# The `--list-initializers` command shows all available initializers. Initializers are referenced by their filename (e.g., `objective_target`, `objective_list`, `simple`) regardless of which subdirectory they're in.
#
# List the available initializers using the --list-initializers flag.

# %%
# !pyrit_scan --list-initializers

# %% [markdown]
# ### Running Scenarios
#
# You need a single scenario to run, you need two things:
#
# 1. A Scenario. Many are defined in `pyrit.scenario.scenarios`. But you can also define your own in initialization_scripts.
# 2. Initializers (which can be supplied via `--initializers` or `--initialization-scripts`). Scenarios often don't need many arguments, but they can be configured in different ways. And at the very least, most need an `objective_target` (the thing you're running a scan against).
# 3. Scenario Strategies (optional). These are supplied by the `--scenario-strategies` flag and tell the scenario what to test, but they are always optional. Also note you can obtain these by running `--list-scenarios`
#
# Basic usage will look something like:
#
# ```shell
# pyrit_scan <scenario> --initializers <initializer1> <initializer2> --scenario-strategies <strategy1> <strategy2>
# ```
#
# You can also override scenario parameters directly from the CLI:
#
# ```shell
# pyrit_scan <scenario> --max-concurrency 10 --max-retries 3 --memory-labels '{"experiment": "test1", "version": "v2"}'
# ```
#
# Or concretely:
#
# ```shell
# !pyrit_scan foundry_scenario --initializers simple openai_objective_target --scenario-strategies base64
# ```
#
# Example with a basic configuration that runs the Foundry scenario against the objective target defined in `openai_objective_target` (which just is an OpenAIChatTarget with `DEFAULT_OPENAI_FRONTEND_ENDPOINT` and `DEFAULT_OPENAI_FRONTEND_KEY`).

# %%
# !pyrit_scan foundry_scenario --initializers openai_objective_target --strategies base64

# %% [markdown]
# Or with all options and multiple initializers and multiple strategies:
#
# ```shell
# pyrit_scan foundry_scenario --database InMemory --initializers simple objective_target objective_list --scenario-strategies easy crescendo
# ```
#
# You can also override scenario execution parameters:
#
# ```shell
# # Override concurrency and retry settings
# pyrit_scan foundry_scenario --initializers simple objective_target --max-concurrency 10 --max-retries 3
#
# # Add custom memory labels for tracking (must be valid JSON)
# pyrit_scan foundry_scenario --initializers simple objective_target --memory-labels '{"experiment": "test1", "version": "v2", "researcher": "alice"}'
# ```
#
# Available CLI parameter overrides:
# - `--max-concurrency <int>`: Maximum number of concurrent attack executions
# - `--max-retries <int>`: Maximum number of automatic retries if the scenario raises an exception
# - `--memory-labels <json>`: Additional labels to apply to all attack runs (must be a JSON string with string keys and values)
#
# You can also use custom initialization scripts by passing file paths. It is relative to your current working directory, but to avoid confusion, full paths are always better:
#
# ```shell
# pyrit_scan encoding_scenario --initialization-scripts ./my_custom_config.py
# ```

# %% [markdown]
# #### Using Custom Scenarios
#
# You can define your own scenarios in initialization scripts. The CLI will automatically discover any `Scenario` subclasses and make them available:
#

from pyrit.common.apply_defaults import apply_defaults

# %%
# my_custom_scenarios.py
from pyrit.scenario import Scenario
from pyrit.scenario.scenario_strategy import ScenarioStrategy


class MyCustomStrategy(ScenarioStrategy):
    """Strategies for my custom scenario."""

    ALL = ("all", {"all"})
    Strategy1 = ("strategy1", set[str]())
    Strategy2 = ("strategy2", set[str]())


@apply_defaults
class MyCustomScenario(Scenario):
    """My custom scenario that does XYZ."""

    @classmethod
    def get_strategy_class(cls):
        return MyCustomStrategy

    @classmethod
    def get_default_strategy(cls):
        return MyCustomStrategy.ALL

    def __init__(self, *, scenario_result_id=None, **kwargs):
        # Scenario-specific configuration only - no runtime parameters
        super().__init__(
            name="My Custom Scenario",
            version=1,
            strategy_class=MyCustomStrategy,
            scenario_result_id=scenario_result_id,
        )
        # ... your scenario-specific initialization code

    async def _get_atomic_attacks_async(self):
        # Build and return your atomic attacks
        return []


# %% [markdown]
# Then discover and run it:
#
# ```shell
# # List to see it's available
# pyrit_scan --list-scenarios --initialization-scripts ./my_custom_scenarios.py
#
# # Run it with parameter overrides
# pyrit_scan my_custom_scenario --initialization-scripts ./my_custom_scenarios.py --max-concurrency 10
# ```
#
# The scenario name is automatically converted from the class name (e.g., `MyCustomScenario` becomes `my_custom_scenario`).
