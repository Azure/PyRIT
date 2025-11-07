# Preloading Datasets for AI RT Scenarios

## Overview

The scenarios in the ai_rt folder class that test datasets be preloaded into PyRIT's `CentralMemory` before running any scenarios. This design ensures that:

1. **Test data is centralized**: All prompts and objectives are stored in a consistent location
2. **Scenarios are reusable**: Multiple scenarios can share the same datasets
3. **Data is retrievable**: The scenario can automatically find and load the correct prompts based on strategy names
4. **Memory is isolated**: Different test runs can use different memory instances (e.g., IN_MEMORY vs SQLite)

## Dataset Naming Schema

The naming schema is **critical** for these scenarios to automatically retrieve the correct datasets. The schema follows this pattern:

```
<dataset_path_prefix><strategy_name>
```

### Components

1. **Dataset Path Prefix** (default: <scenario_name>):
   - Can be customized via the `objective_dataset_path` parameter in the scenario constructor
   - Helps organize datasets in memory when multiple scenario types are being used

2. **Strategy Name** (required):
   - Derived from the strategy enum value
   - Converted to lowercase with underscores (e.g., `HateFictionalStory` → `hate_fictional_story`)
   - Must match exactly for the scenario to find the dataset

### Default Naming Examples for Rapid Response Harm Scenario

| Strategy Enum | Dataset Name |
|--------------|--------------|
| `RapidResponseHarmStrategy.HateFictionalStory` | `rapid_response_harm_hate_fictional_story` |
| `RapidResponseHarmStrategy.FairnessEthnicityInference` | `rapid_response_harm_fairness_ethnicity_inference` |
| `RapidResponseHarmStrategy.ViolenceCivic` | `rapid_response_harm_violence_civic` |
| `RapidResponseHarmStrategy.ViolenceProtestDisruption` | `rapid_response_harm_violence_protest_disruption` |
| `RapidResponseHarmStrategy.SexualContent` | `rapid_response_harm_sexual_content` |
| `RapidResponseHarmStrategy.HarassmentBullying` | `rapid_response_harm_harassment_bullying` |
| `RapidResponseHarmStrategy.MisinformationElection` | `rapid_response_harm_misinformation_election` |
| `RapidResponseHarmStrategy.LeakagePersonalData` | `rapid_response_harm_leakage_personal_data` |

### Custom Dataset Path Prefix

You can customize the prefix when creating a scenario:

```python
scenario = RapidResponseHarmScenario(
    objective_target=my_target,
    adversarial_chat=adversarial_target,
    objective_dataset_path="custom_test_",  # Custom prefix
    scenario_strategies=[RapidResponseHarmStrategy.HateFictionalStory]
)

# Now the dataset name must be: "custom_test_hate_fictional_story"
```


## Common Errors and Solutions

### Error: "No objectives found in the dataset"

**Cause**: The dataset wasn't loaded into memory or the naming doesn't match.

**Solution**:
1. Verify the dataset name matches the strategy name exactly
2. Ensure you called `add_seed_groups_to_memory()` before running the scenario
3. Check that the dataset includes a `SeedObjective` object

```python
# Correct naming
dataset_name = "rapid_response_harm_" + strategy.value  # e.g., "hate_fictional_story"
```

### Error: Dataset not found for custom prefix

**Cause**: The scenario's `objective_dataset_path` doesn't match the dataset names in memory.

**Solution**: Ensure consistency between the scenario configuration and dataset names:

```python
# Scenario configuration
scenario = RapidResponseHarmScenario(
    objective_target=target,
    adversarial_chat=adversarial,
    objective_dataset_path="my_custom_prefix_"  # Must match dataset names
)

# Dataset must be named: "my_custom_prefix_hate_fictional_story"
await create_seed_dataset(
    name="my_custom_prefix_hate_fictional_story",
    prompts=[...],
    objective="..."
)
```

## Additional Resources

- See `rapid_response_harm_scenario.ipynb` for a complete working example
- Check the `RapidResponseHarmStrategy` enum for all available strategies
