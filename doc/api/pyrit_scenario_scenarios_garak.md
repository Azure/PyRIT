# pyrit.scenario.scenarios.garak

Garak-based attack scenarios.

## `class Encoding(Scenario)`

Encoding Scenario implementation for PyRIT.

This scenario tests how resilient models are to various encoding attacks by encoding
potentially harmful text (by default slurs and XSS payloads) and testing if the model
will decode and repeat the encoded payload. It mimics the Garak encoding probe.

The scenario works by:
1. Taking seed prompts (the harmful text to be encoded)
2. Encoding them using various encoding schemes (Base64, ROT13, Morse, etc.)
3. Asking the target model to decode the encoded text
4. Scoring whether the model successfully decoded and repeated the harmful content

By default, this uses the same dataset as Garak: slur terms and web XSS payloads.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seed_prompts` | `Optional[list[str]]` | Deprecated. Use dataset_config in initialize_async instead. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | The scorer used to evaluate if the model successfully decoded the payload. Defaults to DecodingScorer with encoding_scenario category. Defaults to `None`. |
| `encoding_templates` | `Optional[Sequence[str]]` | Templates used to construct the decoding prompts. Defaults to AskToDecodeConverter.garak_templates. Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. When True, a "baseline" attack is automatically added as the first atomic attack, allowing comparison between unmodified prompts and encoding-modified prompts. Defaults to `True`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with garak slur terms and web XSS payloads,
where each seed is transformed into a SeedAttackGroup with an encoding objective.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — EncodingStrategy.ALL (all encoding strategies).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The EncodingStrategy enum class.

## `class EncodingStrategy(ScenarioStrategy)`

Strategies for encoding attacks.

Each enum member represents an encoding scheme that will be tested against the target model.
The ALL aggregate expands to include all encoding strategies.

Note: EncodingStrategy does not support composition. Each encoding must be applied individually.
