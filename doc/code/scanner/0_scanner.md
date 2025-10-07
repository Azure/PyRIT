# PyRIT Scanner

The PyRIT Scanner is a command-line interface (CLI) tool that enables automated security testing and red teaming of generative AI systems through configuration-driven attacks. It provides a declarative way to define and execute complex attack scenarios without writing Python code.

## Overview

The scanner feature allows you to:

- **Configure attacks declaratively** using YAML configuration files
- **Execute multiple attack scenarios** against AI systems
- **Apply prompt converters** to transform and obfuscate prompts
- **Score responses automatically** using various scoring strategies
- **Track and label operations** for organized testing campaigns
- **Run tests locally** with support for future distributed execution

## Key Components

### 1. CLI Command (`pyrit_scan`)

The scanner is invoked through the `pyrit_scan` command, which is automatically available after installing PyRIT:

```bash
python -m pyrit --config-file path/to/config.yaml
```

Or if PyRIT is installed as a package:

```bash
pyrit_scan --config-file path/to/config.yaml
```

### 2. Configuration Files

Scanner behavior is defined entirely through YAML configuration files that specify:

- **Datasets**: Seed prompts to test against the target
- **Scenarios**: Attack strategies to execute (e.g., `RedTeamingAttack`, `PromptSendingAttack`)
- **Targets**: AI systems to test (objective target and optional adversarial target)
- **Converters**: Prompt transformation strategies
- **Scoring**: Automated evaluation of responses
- **Database**: Memory storage configuration
- **Execution Settings**: How to run the scanner (local or distributed)

### 3. Attack Scenarios

The scanner supports multiple attack types:

- **PromptSendingAttack**: Direct prompt sending without multi-turn conversation
- **RedTeamingAttack**: Multi-turn conversational red teaming
- **CrescendoAttack**: Gradual escalation attack strategy
- **TreeOfAttacksWithPruningAttack**: Tree-based exploration with pruning

### 4. Memory and Tracking

All scanner operations are stored in PyRIT's memory system, enabling:

- **Audit trails**: Complete record of all prompts and responses
- **Result analysis**: Post-execution analysis of attack effectiveness
- **Operation labeling**: Tag operations with metadata for organization
- **Export capabilities**: Export results for reporting and analysis

## When to Use the Scanner

The scanner is ideal for:

- **Automated testing pipelines**: CI/CD integration for continuous security testing
- **Batch testing**: Running multiple attack scenarios against various targets
- **Repeatable tests**: Standardized testing with consistent configurations
- **Team collaboration**: Shareable configuration files for consistent testing approaches
- **Quick testing**: Fast execution without writing Python code

## When to Use Python Scripts Instead

For more complex scenarios, consider using PyRIT's Python API directly:

- **Human in the loop**: Red teamers can interact with PyRIT directly in a more hands-on way
- **Custom attack logic**: Implementing new attack strategies
- **Dynamic decision-making**: Conditional logic based on responses
- **Complex workflows**: Orchestrating multiple attacks with dependencies
- **Advanced scoring**: Custom scoring logic not available in standard scorers

## Architecture

The scanner follows this execution flow:

1. **Parse Configuration**: Load and validate YAML configuration
2. **Initialize Database**: Set up memory storage with specified labels
3. **Load Datasets**: Read seed prompts from dataset files
4. **Create Components**: Instantiate targets, converters, and scorers
5. **Execute Scenarios**: Run each configured attack scenario
6. **Score Results**: Apply scoring logic to responses
7. **Print Results**: Display attack outcomes and scores

## Next Steps

- Learn about [configuration options](1_configuration.md) to understand all available settings, including example configurations and usage guidance
- Review sample configurations in the `scanner_configurations/` directory
