# The PyRIT CLI

The PyRIT cli tool that allows you to run automated security testing and red teaming attacks against AI systems using [scenarios](../scenarios/scenarios.ipynb) for strategries and [configuration](../setup/0_configuration.ipynb). 

## Quick Start

For help:

```bash
pyrit_scan --help
```

### Discovery

List all available scenarios:

```bash
pyrit_scan --list-scenarios

Available Scenarios:
================================================================================

  encoding_scenario
    Class: EncodingScenario
    Description:
      Encoding Scenario implementation for PyRIT. This scenario tests how
      resilient models are to various encoding attacks by encoding
      potentially harmful text (by default slurs and XSS payloads) and
      testing if the model will decode and repeat the encoded payload. It
      mimics the Garak encoding probe. The scenario works by: 1. Taking
      seed prompts (the harmful text to be encoded) 2. Encoding them using
      various encoding schemes (Base64, ROT13, Morse, etc.) 3. Asking the
      target model to decode the encoded text 4. Scoring whether the model
      successfully decoded and repeated the harmful content By default,
      this uses the same dataset as Garak: slur terms and web XSS
      payloads.

  foundry_scenario
    Class: FoundryScenario
    Description:
      FoundryScenario is a preconfigured scenario that automatically
      generates multiple AtomicAttack instances based on the specified
      attack strategies. It supports both single-turn attacks (with
      various converters) and multi-turn attacks (Crescendo, RedTeaming),
      making it easy to quickly test a target against multiple attack
      vectors. The scenario can expand difficulty levels (EASY, MODERATE,
      DIFFICULT) into their constituent attack strategies, or you can
      specify individual strategies directly. Note this is not the same as
      the Foundry AI Red Teaming Agent. This is a PyRIT contract so their
      library can make use of PyRIT in a consistent way.

================================================================================

Total scenarios: 2
```

**Tip**: You can also discover user-defined scenarios by providing initialization scripts:

```bash
pyrit_scan --list-scenarios --initialization-scripts ./my_custom_scenarios.py
```

This will load your custom scenario definitions and include them in the list.

List all available built-in initializers:

```bash
pyrit_scan --list-initializers

Available Built-in Initializers:
================================================================================

  simple
    Class: SimpleInitializer
    Name: Simple OpenAI Configuration
    Execution Order: 1
    Required Environment Variables:
      - OPENAI_CHAT_ENDPOINT
      - OPENAI_CHAT_KEY
    Description:
      Basic OpenAI configuration with default converters and scorers.

  airt
    Class: AIRTInitializer
    Name: AIRT Team Configuration
    Execution Order: 1
    Required Environment Variables:
      - AZURE_OPENAI_GPT4O_AAD_ENDPOINT
      - (and several others...)
    Description:
      AIRT team-specific configuration with Azure OpenAI.

  scenarios.objective_target
    Class: ScenarioObjectiveTargetInitializer
    Name: Simple Objective Target Configuration for Scenarios
    Execution Order: 10
    Required Environment Variables:
      - OPENAI_CLI_ENDPOINT
      - OPENAI_CLI_KEY
    Description:
      Sets a default objective_target for scenarios.

  scenarios.objective_list
    Class: ScenarioObjectiveList
    Name: Simple Objective List Configuration for Scenarios
    Execution Order: 10
    Required Environment Variables: None
    Description:
      Sets a default list of objectives for scenarios.

================================================================================

Total initializers: 4
```

### Running Scenarios

You need a single scenario to run, you need two things:

1. A Scenario. Many are defined in `pyrit.scenarios.scenarios`. But you can also define your own in initizliation_scripts.
2. Initializers (which can be supplied via `--initializers` or `--initialization-scripts`). Scenarios often don't need many arguments, but they can be configured in different ways. And at the very least, most need an `objective_target` (the thing you're running a scan against).

Baisc usage will look something like:

```bash
pyrit_scan <scenario> --initializers <initializer1> <initializer2>
```

Example with a basic configuration that runs the Foundry scenario against the objective target defined in `scenarios.objective_target` (which just is an OpenAIChatTarget with `OPENAI_CLI_ENDPOINT` and `OPEN_CLI_KEY`).

```bash
pyrit_scan foundry_scenario --initializers scenarios.objective_target
```

Or with all options and multiple initializers:

```bash
pyrit_scan foundry_scenario --database InMemory --initializers simple scenarios.objective_target scenarios.objective_list
```

You can also use custom initialization scripts by passing file paths:

```bash
pyrit_scan encoding_scenario --initialization-scripts .\my_custom_config.py
```

#### Using Custom Scenarios

You can define your own scenarios in initialization scripts. The CLI will automatically discover any `Scenario` subclasses and make them available:

```python
# my_custom_scenarios.py
from pyrit.scenarios import Scenario
from pyrit.common.apply_defaults import apply_defaults

@apply_defaults
class MyCustomScenario(Scenario):
    """My custom scenario that does XYZ."""
    
    def __init__(self, objective_target=None):
        super().__init__(name="My Custom Scenario", version="1.0")
        self.objective_target = objective_target
        # ... your initialization code
    
    async def initialize_async(self):
        # Load your atomic attacks
        pass
    
    # ... implement other required methods
```

Then discover and run it:

```bash
# List to see it's available
pyrit_scan --list-scenarios --initialization-scripts ./my_custom_scenarios.py

# Run it
pyrit_scan my_custom_scenario --initialization-scripts ./my_custom_scenarios.py
```

The scenario name is automatically converted from the class name (e.g., `MyCustomScenario` becomes `my_custom_scenario`).

## Built-in Initializers

PyRIT includes several built-in initializers you can use with the `--initializers` flag. These will be in `pyrit.setup`, and this list will grow over time.

- **`simple`**: Basic OpenAI configuration with default converters and scorers
- **`airt`**: AIRT team-specific configuration
- **`scenarios.objective_target`**: Almost every scenario needs an objective target. This initializer sets a default OpenAIChatTarget `objective_target` for scenarios
- **`scenarios.objective_list`**: Sets a default list of `objectives` for scenarios

You can also create your own initialization scripts and pass them via `--initialization-scripts` with file paths.


## When to Use the Scanner

The scanner is ideal for:

- **Automated testing pipelines**: CI/CD integration for continuous security testing
- **Batch testing**: Running multiple attack scenarios against various targets
- **Repeatable tests**: Standardized testing with consistent configurations
- **Team collaboration**: Shareable configuration files for consistent testing approaches
- **Quick testing**: Fast execution without writing Python code


## Complete Documentation

For comprehensive documentation about initialization files and setting defaults see:

- **Configuration**: See [configuration.](../doc/code/setup/0_configuration.ipynb)
- **Setting Default Values**: See [default values](../doc/code/setup/default_values.md)
- **Writing Initializers**: See [Initializers](../doc/code/setup/pyrit_initializer.ipynb)

Or visit the [PyRIT documentation website](https://azure.github.io/PyRIT/)