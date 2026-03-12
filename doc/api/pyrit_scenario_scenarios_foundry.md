# pyrit.scenario.scenarios.foundry

Foundry scenario classes.

## `class FoundryScenario(RedTeamAgent)`

Deprecated alias for RedTeamAgent.

This class is deprecated and will be removed in version 0.13.0.
Use `RedTeamAgent` instead.

## `class FoundryStrategy(ScenarioStrategy)`

Strategies for attacks with tag-based categorization.

Each enum member is defined as (value, tags) where:
- value: The strategy name (string)
- tags: Set of tags for categorization (e.g., {"easy", "converter"})

Tags can include complexity levels (easy, moderate, difficult) and other
characteristics (converter, multi_turn, jailbreak, llm_assisted, etc.).

Aggregate tags (EASY, MODERATE, DIFFICULT, ALL) can be used to expand
into all strategies with that tag.

**Methods:**

#### `get_aggregate_tags() → set[str]`

Get the set of tags that represent aggregate categories.

**Returns:**

- `set[str]` — set[str]: Set of tags that are aggregate markers.

#### `supports_composition() → bool`

Indicate that FoundryStrategy supports composition.

**Returns:**

- `bool` — True, as Foundry strategies can be composed together (with rules).

#### `validate_composition(strategies: Sequence[ScenarioStrategy]) → None`

Validate whether the given Foundry strategies can be composed together.

Foundry-specific composition rules:
- Multiple attack strategies (e.g., Crescendo, MultiTurn) cannot be composed together
- Converters can be freely composed with each other
- At most one attack can be composed with any number of converters

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Sequence[ScenarioStrategy]` | The strategies to validate for composition. |

**Raises:**

- `ValueError` — If the composition violates Foundry's rules (e.g., multiple attack).

## `class RedTeamAgent(Scenario)`

RedTeamAgent is a preconfigured scenario that automatically generates multiple
AtomicAttack instances based on the specified attack strategies. It supports both
single-turn attacks (with various converters) and multi-turn attacks (Crescendo,
RedTeaming), making it easy to quickly test a target against multiple attack vectors.

The scenario can expand difficulty levels (EASY, MODERATE, DIFFICULT) into their
constituent attack strategies, or you can specify individual strategies directly.

This scenario is designed for use with the Foundry AI Red Teaming Agent library,
providing a consistent PyRIT contract for their integration.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat` | `Optional[PromptChatTarget]` | Target for multi-turn attacks like Crescendo and RedTeaming. Additionally used for scoring defaults. If not provided, a default OpenAI target will be created using environment variables. Defaults to `None`. |
| `objectives` | `Optional[List[str]]` | Deprecated. Use dataset_config in initialize_async instead. List of attack objectives/prompts to test. Will be removed in a future release. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for attack scoring, including the objective scorer and auxiliary scorers. If not provided, creates a default configuration with a composite scorer using Azure Content Filter and SelfAsk Refusal scorers. Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. When True, a "baseline" attack is automatically added as the first atomic attack, allowing comparison between unmodified prompts and attack-modified prompts. Defaults to `True`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — FoundryStrategy.EASY (easy difficulty strategies).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The FoundryStrategy enum class.
