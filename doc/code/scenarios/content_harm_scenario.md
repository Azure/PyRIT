# Preloading Datasets for End-to-End Scenarios

## Overview

The scenarios in the e2e folder class require that test datasets be preloaded into PyRIT's `CentralMemory` before running any scenarios. This ensures that:

1. **Test data is centralized**: All prompts and objectives are stored in a consistent location
2. **Scenarios are reusable**: Multiple scenarios can share the same datasets
3. **Data is retrievable**: The scenario can automatically find and load the correct prompts based on strategy names
4. **Memory is isolated**: Different test runs can use different memory instances (e.g., IN_MEMORY vs SQLite)

## Sample Datasets

PyRIT provides sample datasets for each harm category in the `ContentHarmStrategy` enum, located in the `pyrit/datasets/seed_prompts/harms/` folder. Each harm category has a corresponding YAML file (e.g., `hate.prompt`, `violence.prompt`, `sexual.prompt`) containing pre-defined prompts and objectives designed to test model behavior for that specific harm type. These datasets can be loaded directly using `SeedDataset.from_yaml_file()` and serve as a starting point for content harm testing, though you can also create custom datasets to suit your specific testing needs.

## Dataset Naming Schema

The naming schema is **critical** for these scenarios to automatically retrieve the correct datasets. The schema follows this pattern:

```
<seed_dataset_prefix>_<strategy_name>
```

### Components

1. **Dataset Path Prefix** (default: <scenario_name>):
   - Can be customized via the `seed_dataset_prefix` parameter in the scenario constructor
   - Helps organize datasets in memory when multiple scenario types are being used

2. **Strategy Name** (required):
   - Derived from the strategy enum value
   - Converted to lowercase with underscores (e.g., `Hate` → `hate`)
   - Must match exactly for the scenario to find the dataset

### Custom Dataset Path Prefix

You can customize the prefix when creating a scenario. For example, in the `ContentHarmScenario`:

```python
scenario = ContentHarmScenario(
    objective_target=my_target,
    adversarial_chat=adversarial_target,
    seed_dataset_prefix="custom_test",  # Custom prefix
    scenario_strategies=[ContentHarmStrategy.Hate]
)

# Now the dataset name must be: "custom_test_hate"
```
## Common Errors and Solutions

### Error: "No objectives found in the dataset"

**Cause**: The dataset wasn't loaded into memory or the naming doesn't match.

**Solution**:
1. Verify the dataset name matches the strategy name exactly
2. Ensure you called `add_seed_groups_to_memory()` before running the scenario
3. Check that the dataset includes a `SeedObjective` object


### Error: Dataset not found for custom prefix

**Cause**: The scenario's `seed_dataset_prefix` doesn't match the dataset names in memory.

**Solution**: Ensure consistency between the scenario configuration and dataset names:

```python
# Scenario configuration
scenario = RapidResponseHarmScenario(
    objective_target=target,
    adversarial_chat=adversarial,
    objective_dataset_path="my_custom_prefix"  # Must match dataset names
)
```

## Additional Resources

- See `content_harm_scenario.ipynb` for a complete working example
- Check the `ContentHarmStrategy` enum for all available strategies
