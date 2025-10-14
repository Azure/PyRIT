# Scanner Configurations

This directory contains sample YAML configuration files for the PyRIT Scanner CLI tool.

## What is the Scanner?

The PyRIT Scanner is a command-line tool that allows you to run automated security testing and red teaming attacks against AI systems using declarative YAML configuration files. Instead of writing Python code, you can define your testing strategy in a configuration file and execute it with a simple command.

## Quick Start

Run a scanner configuration:

```bash
python -m pyrit --config-file scanner_configurations/prompt_send.yaml
```

Or if PyRIT is installed as a package:

```bash
pyrit_scan --config-file scanner_configurations/prompt_send.yaml
```

## Sample Configurations

### `prompt_send.yaml`

Basic direct prompt sending without multi-turn conversation. Good for quick testing and simple scenarios.

### `basic_multi_turn_attack.yaml`

Multi-turn red teaming attack with scoring. Demonstrates adversarial chat configuration and automated response evaluation.

## Complete Documentation

For comprehensive documentation about the scanner feature, including:

- **Architecture and concepts**: See [doc/code/scanner/0_scanner.md](../doc/code/scanner/0_scanner.md)
- **Configuration reference**: See [doc/code/scanner/1_configuration.md](../doc/code/scanner/1_configuration.md)

Or visit the [PyRIT documentation website](https://azure.github.io/PyRIT/).

## Creating Your Own Configurations

Use the sample files as templates. Key sections to configure:

- `datasets`: Paths to seed prompt files
- `scenarios`: Attack strategies (e.g., `RedTeamingAttack`, `PromptSendingAttack`)
- `objective_target`: The AI system to test
- `attack_adversarial_config`: Optional adversarial target for multi-turn attacks
- `converters`: Optional prompt transformations
- `scoring`: Optional automated response evaluation
- `database`: Memory storage configuration
- `execution_settings`: How to run the scanner
