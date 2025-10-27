# Scanner Configuration Reference

This document provides a comprehensive reference for all configuration options available in PyRIT scanner YAML files.

## Configuration File Structure

A scanner configuration file consists of several top-level sections:

```yaml
datasets: [...]           # Required: List of seed prompt files
scenarios: [...]          # Required: Attack scenarios to execute
objective_target: {...}   # Required: Primary target to attack
attack_adversarial_config: {...}  # Optional: Adversarial chat target
converters: [...]         # Optional: Prompt transformations
scoring: {...}            # Optional: Response scoring configuration
database: {...}           # Required: Memory storage configuration
execution_settings: {...} # Optional: Execution environment settings
```

## Datasets

Specifies the seed prompt files to use for testing.

```yaml
datasets:
  - ./pyrit/datasets/seed_prompts/illegal.prompt
  - ./path/to/custom_prompts.yaml
```

- **Type**: List of strings
- **Required**: Yes
- **Description**: Paths to YAML files containing seed prompts. Each file should follow the `SeedDataset` format.

## Scenarios

Defines the attack strategies to execute against the target.

```yaml
scenarios:
  - type: "PromptSendingAttack"
  - type: "RedTeamingAttack"
  - type: "CrescendoAttack"
  - type: "TreeOfAttacksWithPruningAttack"
    depth: 2
```

### Available Scenario Types

#### PromptSendingAttack

Direct prompt sending without multi-turn conversation.

```yaml
- type: "PromptSendingAttack"
```

#### RedTeamingAttack

Multi-turn conversational red teaming with an adversarial target.

```yaml
- type: "RedTeamingAttack"
```

**Requirements**: Must have `attack_adversarial_config` defined.

#### CrescendoAttack

Gradual escalation attack strategy.

```yaml
- type: "CrescendoAttack"
```

**Requirements**: Must have `attack_adversarial_config` defined.

#### TreeOfAttacksWithPruningAttack

Tree-based exploration with pruning for efficient attack discovery.

```yaml
- type: "TreeOfAttacksWithPruningAttack"
  depth: 2  # Optional: Tree depth parameter
```

**Requirements**: Must have `attack_adversarial_config` defined.

### Scenario-Specific Parameters

Some scenarios accept additional parameters that can be specified directly in the configuration:

```yaml
scenarios:
  - type: "TreeOfAttacksWithPruningAttack"
    depth: 3
    width: 5
```

## Additional Parameters

**Important**: All objects in the scanner configuration (attacks, targets, converters, scorers) support additional parameters beyond those explicitly documented here. Any parameter accepted by the underlying Python class constructor can be specified directly in the YAML configuration.

### How It Works

When the scanner instantiates an object, it passes all parameters from the YAML configuration to the class constructor. This means you can use any parameter supported by the class without the scanner needing to know about it explicitly.

### Examples

**Attack with additional parameters**:
```yaml
scenarios:
  - type: "TreeOfAttacksWithPruningAttack"
    depth: 3             # Custom parameter
    width: 5             # Custom parameter
    branching_factor: 2  # Any other parameter the attack accepts
```

**Target with custom configuration**:
```yaml
objective_target:
  type: "OpenAIChatTarget"
  max_tokens: 1000         # Standard OpenAI parameter
  temperature: 0.7         # Standard OpenAI parameter
  frequency_penalty: 0.5   # Any parameter the target's constructor accepts
```

**Scorer with custom settings**:
```yaml
scoring:
  objective_scorer:
    type: "SelfAskTrueFalseScorer"
    true_false_question_path: pyrit/score/config/true_false_question/task_achieved.yaml
    system_prompt: "Custom system prompt"  # If the scorer accepts this parameter
```

**Converter with options**:
```yaml
converters:
  - type: "TranslationConverter"
    language: "french"
    converter_target:
      type: "OpenAIChatTarget"
      temperature: 0.3  # Control randomness in translation
```

### Finding Available Parameters

To discover what parameters are available for a specific class:

1. **Check the class documentation**: Look at the docstring for the `__init__` method
2. **Review the Python code**: Examine the constructor signature in the source code
3. **Refer to API docs**: Check PyRIT's API documentation for the class

The scanner will validate parameters when instantiating objects and provide error messages if invalid parameters are used.

## Objective Target

The primary AI system being tested. This is the target you want to evaluate for vulnerabilities.

```yaml
objective_target:
  type: "OpenAIChatTarget"
  # Additional parameters depend on the target type
```

### Common Target Types

#### OpenAIChatTarget

```yaml
objective_target:
  type: "OpenAIChatTarget"
  # Optional: Override environment variables
  # deployment_name_env_variable: "AZURE_OPENAI_CHAT_DEPLOYMENT"
  # endpoint_env_variable: "AZURE_OPENAI_CHAT_ENDPOINT"
  # api_key_env_variable: "AZURE_OPENAI_CHAT_KEY"
```

#### AzureMLChatTarget

```yaml
objective_target:
  type: "AzureMLChatTarget"
  endpoint_uri: "https://your-endpoint.azureml.net/score"
  api_key: "${AZURE_ML_KEY}"
```

#### HuggingFaceEndpointTarget

```yaml
objective_target:
  type: "HuggingFaceEndpointTarget"
  endpoint: "https://your-hf-endpoint.com"
  token: "${HF_TOKEN}"
```

### Parameter Passing

Any parameter accepted by the target's constructor can be specified in the configuration. The scanner will pass these parameters when instantiating the target.

## Attack Adversarial Config

Optional adversarial chat target used by multi-turn attack scenarios.

```yaml
attack_adversarial_config:
  type: "OpenAIChatTarget"
  # Same parameters as objective_target
```

- **Type**: Target configuration object
- **Required**: Only for multi-turn scenarios (RedTeamingAttack, CrescendoAttack, etc.)
- **Description**: The adversarial target generates attack prompts in multi-turn scenarios.

## Converters

Prompt transformation strategies to obfuscate, encode, or modify prompts before sending.

```yaml
converters:
  - type: "Base64Converter"
  - type: "LeetspeakConverter"
  - type: "ROT13Converter"
  - type: "TranslationConverter"
    language: "french"
```

### Common Converter Types

#### Base64Converter

Encodes prompts in Base64.

```yaml
- type: "Base64Converter"
```

#### LeetspeakConverter

Converts text to leetspeak (e.g., "hello" → "h3ll0").

```yaml
- type: "LeetspeakConverter"
```

#### ROT13Converter

Applies ROT13 cipher to the text.

```yaml
- type: "ROT13Converter"
```

#### TranslationConverter

Translates prompts to another language.

```yaml
- type: "TranslationConverter"
  language: "french"
```

**Note**: May require a `converter_target` for LLM-based translation.

#### LLM-Based Converters

Some converters require an LLM target:

```yaml
- type: "ToneConverter"
  tone: "formal"
  converter_target:
    type: "OpenAIChatTarget"
```

If `converter_target` is not specified, the scanner will automatically use `attack_adversarial_config` for converters that require an LLM.

## Scoring

Configuration for automated scoring of responses.

```yaml
scoring:
  scoring_target:
    type: "OpenAIChatTarget"
  objective_scorer:
    type: "SelfAskTrueFalseScorer"
    true_false_question_path: pyrit/score/config/true_false_question/task_achieved.yaml
```

### Scoring Target

Optional target used specifically for scoring. If not provided, the `attack_adversarial_config` is used.

```yaml
scoring:
  scoring_target:
    type: "OpenAIChatTarget"
```

### Objective Scorer

The scorer evaluates whether the attack achieved its objective.

#### SelfAskTrueFalseScorer

Binary scorer based on true/false questions.

```yaml
objective_scorer:
  type: "SelfAskTrueFalseScorer"
  true_false_question_path: pyrit/score/config/true_false_question/task_achieved.yaml
```

#### SelfAskRefusalScorer

Detects refusals in responses.

```yaml
objective_scorer:
  type: "SelfAskRefusalScorer"
```

#### SelfAskLikertScorer

Scores on a Likert scale.

```yaml
objective_scorer:
  type: "SelfAskLikertScorer"
  likert_scale_path: pyrit/score/config/likert_scales/harmfulness.yaml
```

#### AzureContentFilterScorer

Uses Azure Content Safety API.

```yaml
objective_scorer:
  type: "AzureContentFilterScorer"
```

### Scorer Parameters

Scorers accept various parameters. Use the `Path` type for file paths:

```yaml
objective_scorer:
  type: "SelfAskTrueFalseScorer"
  true_false_question_path: pyrit/score/config/true_false_question/custom_question.yaml
```

## Database

Configuration for memory storage.

```yaml
database:
  type: "SQLite"
  memory_labels:
    operator: "security_team"
    operation: "op_test_campaign"
    test_date: "2025-10-07"
```

### Database Types

#### SQLite (Default)

```yaml
database:
  type: "SQLite"
```

Stores data in the default SQLite database (`dbdata/pyrit.db`).

#### AzureSQL

```yaml
database:
  type: "AzureSQL"
```

Requires Azure SQL configuration in environment variables.

### Memory Labels

Custom key-value pairs to tag all operations in this scan.

```yaml
memory_labels:
  operator: "alice"
  operation: "op_001"
  target_version: "v1.2.3"
  custom_field: "any_value"
```

- **Type**: Dictionary of strings
- **Required**: No
- **Description**: Labels are stored with all memory entries and can be used for filtering and analysis.

## Execution Settings

Configuration for how the scanner runs.

```yaml
execution_settings:
  type: local
  # parallel_nodes: 4  # Future: Number of parallel scenarios
```

### Supported Execution Types

- **local**: Run scenarios sequentially on the local machine (current implementation)
- **azureml**: Distributed execution on Azure ML (planned feature)

### Parameters

- `type`: Execution environment (`local` or `azureml`)
- `parallel_nodes`: Number of scenarios to run in parallel (future feature)

## Complete Example

Here's a comprehensive configuration example:

```yaml
datasets:
  - ./pyrit/datasets/seed_prompts/illegal.prompt
  - ./custom_prompts/security_tests.yaml

scenarios:
  - type: "PromptSendingAttack"
  - type: "RedTeamingAttack"
  - type: "TreeOfAttacksWithPruningAttack"
    depth: 2

objective_target:
  type: "OpenAIChatTarget"
  deployment_name_env_variable: "AZURE_OPENAI_CHAT_DEPLOYMENT"

attack_adversarial_config:
  type: "OpenAIChatTarget"
  deployment_name_env_variable: "AZURE_OPENAI_RED_TEAM_DEPLOYMENT"

converters:
  - type: "Base64Converter"
  - type: "LeetspeakConverter"
  - type: "TranslationConverter"
    language: "french"

scoring:
  scoring_target:
    type: "OpenAIChatTarget"
  objective_scorer:
    type: "SelfAskTrueFalseScorer"
    true_false_question_path: pyrit/score/config/true_false_question/task_achieved.yaml

database:
  type: "SQLite"
  memory_labels:
    operator: "security_team"
    operation: "comprehensive_test"
    date: "2025-10-07"
    version: "1.0"

execution_settings:
  type: local
```

## Environment Variables

The scanner respects PyRIT's standard environment variable configuration. Targets will use default environment variables unless overridden in the configuration:

- `AZURE_OPENAI_CHAT_ENDPOINT`
- `AZURE_OPENAI_CHAT_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- Additional variables specific to other target types

## Validation

The scanner validates configurations using Pydantic models:

- Required fields must be present
- Types must match expected types
- Target classes must exist and be importable
- Dataset files must exist
- Configuration relationships are checked (e.g., multi-turn scenarios require adversarial config)

Invalid configurations will produce clear error messages indicating what needs to be fixed.

## Running the Scanner

### Prerequisites

Before running the scanner, ensure:

1. **PyRIT is installed**: `pip install -e .` from the repository root
2. **Environment file is configured**: Configure API keys and endpoints for your targets via a `.env` file
3. **Dataset files exist**: Verify paths to seed prompt files are correct

### Basic Command

To run the scanner with a configuration file:

```bash
pyrit_scan --config-file path/to/config.yaml
```

## Example Configuration Files

The `scanner_configurations/` directory contains working example configuration files that demonstrate different scanner capabilities. You can use these as templates for your own testing scenarios.

### `prompt_send.yaml`

Simple prompt sending configuration that sends prompts directly to a target without multi-turn conversation. Good for quick testing and basic scenarios.

**Usage**:

```bash
pyrit_scan --config-file scanner_configurations/prompt_send.yaml
```

**What it does**:
- Loads prompts from a dataset file
- Sends each prompt directly to the OpenAI target
- Stores all interactions in the SQLite database
- Prints results to console

### `basic_multi_turn_attack.yaml`

Multi-turn red teaming attack with automated scoring. Demonstrates adversarial chat configuration and response evaluation.

**Usage**:

```bash
pyrit_scan --config-file scanner_configurations/basic_multi_turn_attack.yaml
```

**What it does**:
- Uses an adversarial target to craft sophisticated attack prompts
- Sends attacks to the objective target through multi-turn conversation
- Scores responses using a True/False scorer to determine if objectives were achieved
- Allows for adaptive attack strategies based on target responses

## Creating Custom Configurations

When creating your own scanner configurations:

1. **Start with a sample**: Copy one of the provided configurations from `scanner_configurations/` as a template
2. **Update datasets**: Point to your own seed prompt files or use existing PyRIT datasets
3. **Choose scenarios**: Select attack types that match your testing objectives
4. **Configure targets**: Set up the AI systems you want to test
5. **Add converters** (optional): Include prompt transformations if testing obfuscation resistance
6. **Set up scoring** (optional): Define automated evaluation criteria
7. **Label appropriately**: Use memory labels to organize and track your testing campaigns

Key aspects to customize:

- **Datasets**: Point to your own seed prompt files
- **Scenarios**: Choose attack strategies appropriate for your testing goals
- **Targets**: Configure to test your specific AI systems
- **Converters**: Add prompt transformations to test obfuscation resistance
- **Scoring**: Define how success is measured
- **Memory Labels**: Tag operations with meaningful metadata for tracking

## Analyzing Results

After running the scanner, you can analyze results using PyRIT's memory system:

```python
from pyrit.memory import CentralMemory
from pyrit.setup import initialize_pyrit

# Initialize
initialize_pyrit()
memory = CentralMemory.get_memory_instance()

# Query by operation label
results = memory.get_message_pieces(
    labels={"operation": "your_operation_name"}
)

# Analyze scores
for result in results:
    if result.scores:
        for score in result.scores:
            print(f"Score: {score.score_value}, Rationale: {score.score_rationale}")

# Export results
memory.export_to_json("results.json", labels={"operation": "your_operation_name"})
```

## Best Practices

### Start Simple

Begin with the `prompt_send.yaml` configuration and gradually add complexity as you become familiar with the scanner.

### Use Meaningful Labels

Label operations with descriptive metadata for easy tracking and analysis:

```yaml
memory_labels:
  operator: "alice"
  operation: "sprint_23_security_test"
  date: "2025-10-07"
  target_version: "v2.3.1"
```

### Organize Configuration Files

Structure your configurations by purpose (production testing, development, research, etc.).

### Version Control Configurations

Store scanner configurations in version control alongside your code to track changes and enable team collaboration.

## Troubleshooting

### Configuration Validation Errors

If you see validation errors like `ValueError: Scenario 'type' must not be empty`, check that all required fields are present and properly formatted.

### Target Connection Issues

If targets fail to connect:

1. Verify environment variables are set
2. Check network connectivity
3. Confirm API keys are valid
4. Review target-specific configuration

### Out of Memory

For large scans:

1. Process datasets in smaller batches
2. Use fewer scenarios per run
3. Consider reducing converter chains

### Slow Execution

To improve performance:

1. Use faster models for adversarial/scoring targets
2. Reduce tree depth for TreeOfAttacks
3. Limit the number of prompts in datasets
4. Optimize converter selection
