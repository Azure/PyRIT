# pyrit.scenario

High-level scenario classes for running attack configurations.

## `class AtomicAttack`

Represents a single atomic attack test combining an attack strategy and dataset.

An AtomicAttack is an executable unit that executes a configured attack against
all objectives in a dataset. Multiple AtomicAttacks can be grouped together into
larger test scenarios for comprehensive security testing and evaluation.

The AtomicAttack uses SeedAttackGroups as the single source of truth for objectives,
prepended conversations, and next messages. Each SeedAttackGroup must have an objective set.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `atomic_attack_name` | `str` | Used to group an AtomicAttack with related attacks for a strategy. |
| `attack` | `AttackStrategy` | The configured attack strategy to execute. |
| `seed_groups` | `List[SeedAttackGroup]` | List of seed attack groups. Each seed group must have an objective set. The seed groups serve as the single source of truth for objectives, prepended conversations, and next messages. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | Optional chat target for generating adversarial prompts or simulated conversations. Required when seed groups contain SeedSimulatedConversation configurations. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Optional scorer for evaluating simulated conversations. Required when seed groups contain SeedSimulatedConversation configurations. Defaults to `None`. |
| `memory_labels` | `Optional[Dict[str, str]]` | Additional labels to apply to prompts. These labels help track and categorize the atomic attack in memory. Defaults to `None`. |
| `**attack_execute_params` | `Any` | Additional parameters to pass to the attack execution method (e.g., batch_size). Defaults to `{}`. |

**Methods:**

#### `filter_seed_groups_by_objectives(remaining_objectives: list[str]) → None`

Filter seed groups to only those with objectives in the remaining list.

This is used for scenario resumption to skip already completed objectives.

| Parameter | Type | Description |
|---|---|---|
| `remaining_objectives` | `List[str]` | List of objectives that still need to be executed. |

#### run_async

```python
run_async(max_concurrency: int = 1, return_partial_on_failure: bool = True, attack_params: Any = {}) → AttackExecutorResult[AttackResult]
```

Execute the atomic attack against all seed groups.

This method uses AttackExecutor to run the configured attack against
all seed groups.

When return_partial_on_failure=True (default), this method will return
an AttackExecutorResult containing both completed results and incomplete
objectives (those that didn't finish execution due to exceptions). This allows
scenarios to save progress and retry only the incomplete objectives.

Note: "completed" means the execution finished, not that the attack objective
was achieved. "incomplete" means execution didn't finish (threw an exception).

| Parameter | Type | Description |
|---|---|---|
| `max_concurrency` | `int` | Maximum number of concurrent attack executions. Defaults to 1 for sequential execution. Defaults to `1`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results even when some objectives don't complete execution. If False, raises an exception on any execution failure. Defaults to True. Defaults to `True`. |
| `**attack_params` | `Any` | Additional parameters to pass to the attack strategy. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackResult]` — AttackExecutorResult[AttackResult]: Result containing completed attack results and
incomplete objectives (those that didn't finish execution).

**Raises:**

- `ValueError` — If the attack execution fails completely and return_partial_on_failure=False.

## `class DatasetConfiguration`

Configuration for scenario datasets.

This class provides a unified way to specify the dataset source for scenarios.
Only ONE of `seed_groups` or `dataset_names` can be set.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seed_groups` | `Optional[List[SeedGroup]]` | Explicit list of SeedGroup to use. Defaults to `None`. |
| `dataset_names` | `Optional[List[str]]` | Names of datasets to load from memory. Defaults to `None`. |
| `max_dataset_size` | `Optional[int]` | If set, randomly samples up to this many SeedGroups (without replacement). Defaults to `None`. |
| `scenario_composites` | `Optional[Sequence[ScenarioCompositeStrategy]]` | The scenario strategies being executed. Subclasses can use this to filter or customize which seed groups are loaded. Defaults to `None`. |

**Methods:**

#### `get_all_seed_attack_groups() → list[SeedAttackGroup]`

Resolve and return all seed groups as SeedAttackGroups in a flat list.

This is a convenience method that calls get_seed_attack_groups() and flattens
the results into a single list. Use this for attack scenarios that need
SeedAttackGroup functionality.

**Returns:**

- `list[SeedAttackGroup]` — List[SeedAttackGroup]: All resolved seed attack groups from all datasets.

**Raises:**

- `ValueError` — If no seed groups could be resolved from the configuration.

#### `get_all_seed_groups() → list[SeedGroup]`

Resolve and return all seed groups as a flat list.

This is a convenience method that calls get_seed_groups() and flattens
the results into a single list. Use this when you don't need to track
which dataset each seed group came from.

**Returns:**

- `list[SeedGroup]` — List[SeedGroup]: All resolved seed groups from all datasets,
with max_dataset_size applied per dataset.

**Raises:**

- `ValueError` — If no seed groups could be resolved from the configuration.

#### `get_all_seeds() → list[Seed]`

Load all seed prompts from memory for all configured datasets.

This is a convenience method that retrieves SeedPrompt objects directly
from memory for all configured datasets. If max_dataset_size is set, randomly
samples up to that many prompts per dataset (without replacement).

**Returns:**

- `list[Seed]` — List[SeedPrompt]: List of SeedPrompt objects from all configured datasets.
Returns an empty list if no prompts are found.

**Raises:**

- `ValueError` — If no dataset names are configured.

#### `get_default_dataset_names() → list[str]`

Get the list of default dataset names for this configuration.

This is used by the CLI to display what datasets the scenario uses by default.

**Returns:**

- `list[str]` — List[str]: List of dataset names, or empty list if using explicit seed_groups.

#### `get_seed_attack_groups() → dict[str, list[SeedAttackGroup]]`

Resolve and return seed groups as SeedAttackGroups, grouped by dataset.

This wraps get_seed_groups() and converts each SeedGroup to a SeedAttackGroup.
Use this when you need attack-specific functionality like objectives,
prepended conversations, or simulated conversation configuration.

**Returns:**

- `dict[str, list[SeedAttackGroup]]` — Dict[str, List[SeedAttackGroup]]: Dictionary mapping dataset names to their
seed attack groups.

**Raises:**

- `ValueError` — If no seed groups could be resolved from the configuration.

#### `get_seed_groups() → dict[str, list[SeedGroup]]`

Resolve and return seed groups based on the configuration.

This method handles all resolution logic:
1. If seed_groups is set, use those directly (under key '_explicit_seed_groups')
2. If dataset_names is set, load from memory using those names

In all cases, max_dataset_size is applied **per dataset** if set.

Subclasses can override this to filter or customize which seed groups
are loaded based on the stored scenario_composites.

**Returns:**

- `dict[str, list[SeedGroup]]` — Dict[str, List[SeedGroup]]: Dictionary mapping dataset names to their
seed groups. When explicit seed_groups are provided, the key is
'_explicit_seed_groups'. Each dataset's seed groups are potentially
sampled down to max_dataset_size.

**Raises:**

- `ValueError` — If no seed groups could be resolved from the configuration.

#### `has_data_source() → bool`

Check if this configuration has a data source configured.

**Returns:**

- `bool` — True if seed_groups or dataset_names is configured.

## `class Scenario(ABC)`

Groups and executes multiple AtomicAttack instances sequentially.

A Scenario represents a comprehensive testing campaign composed of multiple
atomic attack tests (AtomicAttacks). It executes each AtomicAttack in sequence and
aggregates the results into a ScenarioResult.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Descriptive name for the scenario. Defaults to `''`. |
| `version` | `int` | Version number of the scenario. |
| `strategy_class` | `Type[ScenarioStrategy]` | The strategy enum class for this scenario. |
| `objective_scorer` | `Scorer` | The objective scorer used to evaluate attack results. |
| `include_default_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Most scenarios should have some kind of baseline so users can understand the impact of strategies, but subclasses can optionally write their own custom baselines. Defaults to True. Defaults to `True`. |
| `scenario_result_id` | `Optional[Union[uuid.UUID, str]]` | Optional ID of an existing scenario result to resume. Can be either a UUID object or a string representation of a UUID. If provided and found in memory, the scenario will resume from prior progress. All other parameters must still match the stored scenario configuration. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

This abstract method must be implemented by all scenario subclasses to return
a DatasetConfiguration specifying the default datasets to use when no
dataset_config is provided by the user.

**Returns:**

- `DatasetConfiguration` — The default dataset configuration.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

This abstract method must be implemented by all scenario subclasses to return
the default aggregate strategy (like EASY, ALL) used when scenario_strategies
parameter is None.

**Returns:**

- `ScenarioStrategy` — The default aggregate strategy (e.g., FoundryStrategy.EASY, EncodingStrategy.ALL).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

This abstract method must be implemented by all scenario subclasses to return
the ScenarioStrategy enum class that defines the available attack strategies
for the scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The strategy enum class (e.g., FoundryStrategy, EncodingStrategy).

#### initialize_async

```python
initialize_async(objective_target: PromptTarget = REQUIRED_VALUE, scenario_strategies: Optional[Sequence[ScenarioStrategy | ScenarioCompositeStrategy]] = None, dataset_config: Optional[DatasetConfiguration] = None, max_concurrency: int = 10, max_retries: int = 0, memory_labels: Optional[dict[str, str]] = None) → None
```

Initialize the scenario by populating self._atomic_attacks and creating the ScenarioResult.

This method allows scenarios to be initialized with atomic attacks after construction,
which is useful when atomic attacks require async operations to be built.

If a scenario_result_id was provided in __init__, this method will check if it exists
in memory and validate that the stored scenario matches the current configuration.
If it matches, the scenario will resume from prior progress. If it doesn't match or
doesn't exist, a new scenario result will be created.

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `scenario_strategies` | `Optional[Sequence[ScenarioStrategy | ScenarioCompositeStrategy]]` |  The strategies to execute. Can be a list of bare ScenarioStrategy enums or ScenarioCompositeStrategy instances for advanced composition. Bare enums are automatically wrapped into composites. If None, uses the default aggregate from the scenario's configuration. Defaults to `None`. |
| `dataset_config` | `Optional[DatasetConfiguration]` | Configuration for the dataset source. Use this to specify dataset names or maximum dataset size from the CLI. If not provided, scenarios use their default_dataset_config(). Defaults to `None`. |
| `max_concurrency` | `int` | Maximum number of concurrent attack executions. Defaults to 1. Defaults to `10`. |
| `max_retries` | `int` | Maximum number of automatic retries if the scenario raises an exception. Set to 0 (default) for no automatic retries. If set to a positive number, the scenario will automatically retry up to this many times after an exception. For example, max_retries=3 allows up to 4 total attempts (1 initial + 3 retries). Defaults to `0`. |
| `memory_labels` | `Optional[Dict[str, str]]` | Additional labels to apply to all attack runs in the scenario. These help track and categorize the scenario. Defaults to `None`. |

**Raises:**

- `ValueError` — If no objective_target is provided.

#### `run_async() → ScenarioResult`

Execute all atomic attacks in the scenario sequentially.

Each AtomicAttack is executed in order, and all results are aggregated
into a ScenarioResult containing the scenario metadata and all attack results.
This method supports resumption - if the scenario raises an exception partway through,
calling run_async again will skip already-completed objectives.

If max_retries is set, the scenario will automatically retry after an exception up to
the specified number of times. Each retry will resume from where it left off,
skipping completed objectives.

**Returns:**

- `ScenarioResult` — Contains scenario identifier and aggregated list of all
attack results from all atomic attacks.

**Raises:**

- `ValueError` — If the scenario has no atomic attacks configured. If your scenario
requires initialization, call await scenario.initialize() first.
- `ValueError` — If the scenario raises an exception after exhausting all retry attempts.
- `RuntimeError` — If the scenario fails for any other reason while executing.

## `class ScenarioCompositeStrategy`

Represents a composition of one or more attack strategies.

This class encapsulates a collection of ScenarioStrategy instances along with
an auto-generated descriptive name, making it easy to represent both single strategies
and composed multi-strategy attacks.

The name is automatically derived from the strategies:
- Single strategy: Uses the strategy's value (e.g., "base64")
- Multiple strategies: Generates "ComposedStrategy(base64, rot13)"

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Sequence[ScenarioStrategy]` | The sequence of strategies in this composition. Must contain at least one strategy. |

**Methods:**

#### extract_single_strategy_values

```python
extract_single_strategy_values(composites: Sequence[ScenarioCompositeStrategy], strategy_type: type[T]) → set[str]
```

Extract strategy values from single-strategy composites.

This is a helper method for scenarios that don't support composition and need
to filter or map strategies by their values. It flattens the composites into
a simple set of strategy values.

This method enforces that all composites contain only a single strategy. If any
composite contains multiple strategies, a ValueError is raised.

| Parameter | Type | Description |
|---|---|---|
| `composites` | `Sequence[ScenarioCompositeStrategy]` | List of composite strategies. Each composite must contain only a single strategy. |
| `strategy_type` | `type[T]` | The strategy enum type to filter by. |

**Returns:**

- `set[str]` — Set[str]: Set of strategy values (e.g., {"base64", "rot13", "morse_code"}).

**Raises:**

- `ValueError` — If any composite contains multiple strategies.

#### `get_composite_name(strategies: Sequence[ScenarioStrategy]) → str`

Generate a descriptive name for a composition of strategies.

For single strategies, returns the strategy's value.
For multiple strategies, generates a name like "ComposedStrategy(base64, rot13)".

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Sequence[ScenarioStrategy]` | The strategies to generate a name for. |

**Returns:**

- `str` — The generated composite name.

**Raises:**

- `ValueError` — If strategies is empty.

#### normalize_compositions

```python
normalize_compositions(compositions: list[ScenarioCompositeStrategy], strategy_type: type[T]) → list[ScenarioCompositeStrategy]
```

Example::

    # Aggregate expands to individual strategies
    [ScenarioCompositeStrategy(strategies=[EASY])]
    -> [ScenarioCompositeStrategy(strategies=[Base64]),
        ScenarioCompositeStrategy(strategies=[ROT13]), ...]

    # Concrete composition preserved
    [ScenarioCompositeStrategy(strategies=[Base64, Atbash])]
    -> [ScenarioCompositeStrategy(strategies=[Base64, Atbash])]

    # Error: Cannot mix aggregate with concrete in same composition
    [ScenarioCompositeStrategy(strategies=[EASY, Base64])] -> ValueError

| Parameter | Type | Description |
|---|---|---|
| `compositions` | `List[ScenarioCompositeStrategy]` | List of composite strategies to normalize. |
| `strategy_type` | `type[T]` | The strategy enum type to use for normalization and validation. |

**Returns:**

- `list[ScenarioCompositeStrategy]` — List[ScenarioCompositeStrategy]: Normalized list of composite strategies with aggregates expanded.

**Raises:**

- `ValueError` — If compositions is empty, contains empty compositions,
mixes aggregates with concrete strategies in the same composition,
has multiple aggregates in one composition, or violates validate_composition() rules.

## `class ScenarioIdentifier`

Scenario result class for aggregating results from multiple AtomicAttacks.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Name of the scenario. |
| `description` | `str` | Description of the scenario. Defaults to `''`. |
| `scenario_version` | `int` | Version of the scenario. Defaults to `1`. |
| `init_data` | `Optional[dict]` | Initialization data. Defaults to `None`. |
| `pyrit_version` | `Optional[str]` | PyRIT version string. If None, uses current version. Defaults to `None`. |

## `class ScenarioResult`

Scenario result class for aggregating scenario results.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `scenario_identifier` | `ScenarioIdentifier` | Identifier for the executed scenario. |
| `objective_target_identifier` | `Union[Dict[str, Any], TargetIdentifier]` | Target identifier. |
| `attack_results` | `dict[str, List[AttackResult]]` | Results grouped by atomic attack name. |
| `objective_scorer_identifier` | `Union[Dict[str, Any], ScorerIdentifier]` | Objective scorer identifier. |
| `scenario_run_state` | `ScenarioRunState` | Current scenario run state. Defaults to `'CREATED'`. |
| `labels` | `Optional[dict[str, str]]` | Optional labels. Defaults to `None`. |
| `completion_time` | `Optional[datetime]` | Optional completion timestamp. Defaults to `None`. |
| `number_tries` | `int` | Number of run attempts. Defaults to `0`. |
| `id` | `Optional[uuid.UUID]` | Optional scenario result ID. Defaults to `None`. |
| `objective_scorer` | `Optional[Scorer]` | Deprecated scorer object parameter. Defaults to `None`. |

**Methods:**

#### `get_objectives(atomic_attack_name: Optional[str] = None) → list[str]`

Get the list of unique objectives for this scenario.

| Parameter | Type | Description |
|---|---|---|
| `atomic_attack_name` | `Optional[str]` | Name of specific atomic attack to include. If None, includes objectives from all atomic attacks. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[str]` — List[str]: Deduplicated list of objectives.

#### `get_scorer_evaluation_metrics() → Optional[ScorerMetrics]`

Get the evaluation metrics for the scenario's scorer from the scorer evaluation registry.

**Returns:**

- `Optional[ScorerMetrics]` — The evaluation metrics object, or None if not found.

#### `get_strategies_used() → list[str]`

Get the list of strategies used in this scenario.

**Returns:**

- `list[str]` — List[str]: Atomic attack strategy names present in the results.

#### `normalize_scenario_name(scenario_name: str) → str`

Normalize a scenario name to match the stored class name format.

Converts CLI-style snake_case names (e.g., "foundry" or "content_harms") to
PascalCase class names (e.g., "Foundry" or "ContentHarms") for database queries.
If the input is already in PascalCase or doesn't match the snake_case pattern,
it is returned unchanged.

This is the inverse of ScenarioRegistry._class_name_to_scenario_name().

| Parameter | Type | Description |
|---|---|---|
| `scenario_name` | `str` | The scenario name to normalize. |

**Returns:**

- `str` — The normalized scenario name suitable for database queries.

#### `objective_achieved_rate(atomic_attack_name: Optional[str] = None) → int`

Get the success rate of this scenario.

| Parameter | Type | Description |
|---|---|---|
| `atomic_attack_name` | `Optional[str]` | Name of specific atomic attack to calculate rate for. If None, calculates rate across all atomic attacks. Defaults to None. Defaults to `None`. |

**Returns:**

- `int` — Success rate as a percentage (0-100).

## `class ScenarioStrategy(Enum)`

Base class for attack strategies with tag-based categorization and aggregation.

This class provides a pattern for defining attack strategies as enums where each
strategy has a set of tags for flexible categorization. It supports aggregate tags
(like "easy", "moderate", "difficult" or "fast", "medium") that automatically expand
to include all strategies with that tag.

**Tags**: Flexible categorization system where strategies can have multiple tags
(e.g., {"easy", "converter"}, {"difficult", "multi_turn"})

Subclasses should define their enum members with (value, tags) tuples and
override the get_aggregate_tags() classmethod to specify which tags
represent aggregates that should expand.

**Convention**: All subclasses should include `ALL = ("all", {"all"})` as the first
aggregate member. The base class automatically handles expanding "all" to
include all non-aggregate strategies.

The normalization process automatically:
1. Expands aggregate tags into their constituent strategies
2. Excludes the aggregate tag enum members themselves from the final set
3. Handles the special "all" tag by expanding to all non-aggregate strategies

**Methods:**

#### `get_aggregate_strategies() → list[T]`

Get all aggregate strategies for this strategy enum.

This method returns only the aggregate markers (like ALL, EASY, MODERATE, DIFFICULT)
that are used to group concrete strategies by tags.

**Returns:**

- `list[T]` — list[T]: List of all aggregate strategies.

#### `get_aggregate_tags() → set[str]`

Get the set of tags that represent aggregate categories.

Subclasses should override this method to specify which tags
are aggregate markers (e.g., {"easy", "moderate", "difficult"} for complexity-based
scenarios or {"fast", "medium"} for speed-based scenarios).

The base class automatically includes "all" as an aggregate tag that expands
to all non-aggregate strategies.

**Returns:**

- `set[str]` — Set[str]: Set of tags that represent aggregates.

#### `get_all_strategies() → list[T]`

Get all non-aggregate strategies for this strategy enum.

This method returns all concrete attack strategies, excluding aggregate markers
(like ALL, EASY, MODERATE, DIFFICULT) that are used for grouping.

**Returns:**

- `list[T]` — list[T]: List of all non-aggregate strategies.

#### `get_strategies_by_tag(tag: str) → set[T]`

Get all attack strategies that have a specific tag.

This method returns concrete attack strategies (not aggregate markers)
that include the specified tag.

| Parameter | Type | Description |
|---|---|---|
| `tag` | `str` | The tag to filter by (e.g., "easy", "converter", "multi_turn"). |

**Returns:**

- `set[T]` — Set[T]: Set of strategies that include the specified tag, excluding
    any aggregate markers.

#### `normalize_strategies(strategies: set[T]) → set[T]`

Normalize a set of attack strategies by expanding aggregate tags.

This method processes a set of strategies and expands any aggregate tags
(like EASY, MODERATE, DIFFICULT or FAST, MEDIUM) into their constituent concrete strategies.
The aggregate tag markers themselves are removed from the result.

The special "all" tag is automatically supported and expands to all non-aggregate strategies.

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Set[T]` | The initial set of attack strategies, which may include                 aggregate tags. |

**Returns:**

- `set[T]` — Set[T]: The normalized set of concrete attack strategies with aggregate tags
   expanded and removed.

#### prepare_scenario_strategies

```python
prepare_scenario_strategies(strategies: Sequence[T | ScenarioCompositeStrategy] | None = None, default_aggregate: T | None = None) → list[ScenarioCompositeStrategy]
```

Prepare and normalize scenario strategies for use in a scenario.

This helper method simplifies scenario initialization by:
1. Handling None input with sensible defaults
2. Auto-wrapping bare ScenarioStrategy instances into ScenarioCompositeStrategy
3. Expanding aggregate tags (like EASY, ALL) into concrete strategies
4. Validating compositions according to the strategy's rules

This eliminates boilerplate code in scenario __init__ methods.

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Sequence[T | ScenarioCompositeStrategy] | None` | The strategies to prepare. Can be a mix of bare strategy enums and composite strategies. If None, uses default_aggregate to determine defaults. If an empty sequence, returns an empty list (useful for baseline-only execution). Defaults to `None`. |
| `default_aggregate` | `T | None` | The aggregate strategy to use when strategies is None. Common values: MyStrategy.ALL, MyStrategy.EASY. If None when strategies is None, raises ValueError. Defaults to `None`. |

**Returns:**

- `list[ScenarioCompositeStrategy]` — List[ScenarioCompositeStrategy]: Normalized list of composite strategies ready for use.
May be empty if an empty sequence was explicitly provided.

**Raises:**

- `ValueError` — If strategies is None and default_aggregate is None, or if compositions
       are invalid according to validate_composition().

#### `supports_composition() → bool`

Indicate whether this strategy type supports composition.

By default, strategies do NOT support composition (only single strategies allowed).
Subclasses that support composition (e.g., FoundryStrategy) should override this
to return True and implement validate_composition() to enforce their specific rules.

**Returns:**

- `bool` — True if composition is supported, False otherwise.

#### `validate_composition(strategies: Sequence[T]) → None`

Validate whether the given strategies can be composed together.

The base implementation checks supports_composition() and raises an error if
composition is not supported and multiple strategies are provided.

Subclasses that support composition should override this method to define their
specific composition rules (e.g., "no more than one attack strategy").

| Parameter | Type | Description |
|---|---|---|
| `strategies` | `Sequence[T]` | The strategies to validate for composition. |

**Raises:**

- `ValueError` — If the composition is invalid according to the subclass's rules.
        The error message should clearly explain what rule was violated.
