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
# # The PyRIT CLI
#
# The PyRIT cli tool that allows you to run automated security testing and red teaming attacks against AI systems using [scenarios](../scenarios/scenarios.ipynb) for strategies and [configuration](../setup/0_configuration.ipynb).
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
# 1. A Scenario. Many are defined in `pyrit.scenarios.scenarios`. But you can also define your own in initialization_scripts.
# 2. Initializers (which can be supplied via `--initializers` or `--initialization-scripts`). Scenarios often don't need many arguments, but they can be configured in different ways. And at the very least, most need an `objective_target` (the thing you're running a scan against).
# 3. Scenario Strategies (optional). These are supplied by the `--scenario-strategies` flag and tell the scenario what to test, but they are always optional. Also note you can obtain these by running `--list-scenarios`
#
# Basic usage will look something like:
#
# ```shell
# pyrit_scan <scenario> --initializers <initializer1> <initializer2> --scenario-strategies <stragegy1> <strategy2>
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
# !pyrit_scan foundry_scenario --initializers openai_objective_target --scenario-strategies base64

# %% [markdown]
# Or with all options and multiple initializers and multiple strategies:
#
# ```shell
# pyrit_scan foundry_scenario --database InMemory --initializers simple objective_target objective_list --scenario-strategies easy crescendo
# ```
#
# You can also use custom initialization scripts by passing file paths. It is relative to your current working directory, but to avoid confusion, full paths are always better:
#
# ```shell
# pyrit_scan encoding_scenario --initialization-scripts ./my_custom_config.py
# ```
#
# #### Using Custom Scenarios
#
# You can define your own scenarios in initialization scripts. The CLI will automatically discover any `Scenario` subclasses and make them available:
#
# ```python
# # my_custom_scenarios.py
# from pyrit.scenarios import Scenario
# from pyrit.common.apply_defaults import apply_defaults
#
# @apply_defaults
# class MyCustomScenario(Scenario):
#     """My custom scenario that does XYZ."""
#
#     def __init__(self, objective_target=None):
#         super().__init__(name="My Custom Scenario", version="1.0")
#         self.objective_target = objective_target
#         # ... your initialization code
#
#     async def initialize_async(self):
#         # Load your atomic attacks
#         pass
#
#     # ... implement other required methods
# ```
#
# Then discover and run it:
#
# ```shell
# # List to see it's available
# pyrit_scan --list-scenarios --initialization-scripts ./my_custom_scenarios.py
#
# # Run it
# pyrit_scan my_custom_scenario --initialization-scripts ./my_custom_scenarios.py
# ```
#
# The scenario name is automatically converted from the class name (e.g., `MyCustomScenario` becomes `my_custom_scenario`).
#
#
# ## When to Use the Scanner
#
# The scanner is ideal for:
#
# - **Automated testing pipelines**: CI/CD integration for continuous security testing
# - **Batch testing**: Running multiple attack scenarios against various targets
# - **Repeatable tests**: Standardized testing with consistent configurations
# - **Team collaboration**: Shareable configuration files for consistent testing approaches
# - **Quick testing**: Fast execution without writing Python code
#
#
# ## Complete Documentation
#
# For comprehensive documentation about initialization files and setting defaults see:
#
# - **Configuration**: See [configuration](../setup/0_configuration.ipynb)
# - **Setting Default Values**: See [default values](../setup/default_values.md)
# - **Writing Initializers**: See [Initializers](../setup/pyrit_initializer.ipynb)
#
# Or visit the [PyRIT documentation website](https://azure.github.io/PyRIT/)
