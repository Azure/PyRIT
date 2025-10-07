# Scanner Usage Examples

This document provides practical guidance on using the PyRIT scanner for various testing scenarios.

## Basic Usage

### Running the Scanner

To run the scanner with a configuration file:

```bash
python -m pyrit --config-file scanner_configurations/basic_multi_turn_attack.yaml
```

Or if PyRIT is installed as a package:

```bash
pyrit_scan --config-file scanner_configurations/basic_multi_turn_attack.yaml
```

### Prerequisites

Before running the scanner, ensure:

1. **PyRIT is installed**: `pip install -e .` from the repository root
2. **Environment file is configured**: Configure API keys and endpoints for your targets via a `.env` file
3. **Dataset files exist**: Verify paths to seed prompt files are correct

## Sample Configurations

The `scanner_configurations/` directory contains example configuration files that demonstrate different scanner capabilities:

### `prompt_send.yaml`

Simple prompt sending configuration that sends prompts directly to a target without multi-turn conversation. Good for quick testing and basic scenarios.

**Usage**:

```bash
python -m pyrit --config-file scanner_configurations/prompt_send.yaml
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
python -m pyrit --config-file scanner_configurations/basic_multi_turn_attack.yaml
```

**What it does**:
- Uses an adversarial target to craft sophisticated attack prompts
- Sends attacks to the objective target through multi-turn conversation
- Scores responses using a True/False scorer to determine if objectives were achieved
- Allows for adaptive attack strategies based on target responses

## Customizing Configurations

You can use these sample files as templates for your own testing scenarios. Key aspects to customize:

- **Datasets**: Point to your own seed prompt files
- **Scenarios**: Choose attack strategies appropriate for your testing goals
- **Targets**: Configure to test your specific AI systems
- **Converters**: Add prompt transformations to test obfuscation resistance
- **Scoring**: Define how success is measured
- **Memory Labels**: Tag operations with meaningful metadata for tracking

## Creating Your Own Configurations

When creating custom scanner configurations:

1. **Start with a sample**: Copy one of the provided configurations as a template
2. **Update datasets**: Point to your own seed prompt files or use existing PyRIT datasets
3. **Choose scenarios**: Select attack types that match your testing objectives
4. **Configure targets**: Set up the AI systems you want to test
5. **Add converters** (optional): Include prompt transformations if testing obfuscation resistance
6. **Set up scoring** (optional): Define automated evaluation criteria
7. **Label appropriately**: Use memory labels to organize and track your testing campaigns

## Analyzing Results

After running the scanner, you can analyze results using PyRIT's memory system:

```python
from pyrit.memory import CentralMemory
from pyrit.common import initialize_pyrit

# Initialize
initialize_pyrit()
memory = CentralMemory.get_memory_instance()

# Query by operation label
results = memory.get_prompt_request_pieces(
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

### 1. Start Simple

Begin with the `prompt_send.yaml` configuration and gradually add complexity as you become familiar with the scanner.

### 2. Use Meaningful Labels

Label operations with descriptive metadata for easy tracking and analysis:

```yaml
memory_labels:
  operator: "alice"
  operation: "sprint_23_security_test"
  date: "2025-10-07"
  target_version: "v2.3.1"
```

### 3. Organize Configuration Files

Structure your configurations by purpose (production testing, development, research, etc.).

### 4. Version Control Configurations

Store scanner configurations in version control alongside your code to track changes and enable team collaboration.

## Troubleshooting

### Configuration Validation Errors

If you see validation errors:

```
ValueError: Scenario 'type' must not be empty.
```

Check that all required fields are present and properly formatted.

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

## Next Steps

- Review the [configuration reference](1_configuration.md) for all available options
- Explore the [scanner overview](0_scanner.md) for architectural details
- Check sample configurations in `scanner_configurations/`
