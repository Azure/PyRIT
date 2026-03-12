# pyrit.executor.attack.core

Core attack strategy module.

## `class AttackAdversarialConfig`

Adversarial configuration for attacks that involve adversarial chat targets.

This class defines the configuration for attacks that utilize an adversarial chat target,
including the target chat model, system prompt, and seed prompt for the attack.

## `class AttackContext(StrategyContext, ABC, Generic[AttackParamsT])`

Base class for all attack contexts.

This class holds both the immutable attack parameters and the mutable
execution state. The params field contains caller-provided inputs,
while other fields track execution progress.

Attacks that generate certain values internally (e.g., RolePlayAttack generates
next_message and prepended_conversation) can set the mutable override fields
(_next_message_override, _prepended_conversation_override) during _setup_async.

## `class AttackConverterConfig(StrategyConverterConfig)`

Configuration for prompt converters used in attacks.

This class defines the converter configurations that transform prompts
during the attack process, both for requests and responses.

## `class AttackExecutor`

Manages the execution of attack strategies with support for parallel execution.

The AttackExecutor provides controlled execution of attack strategies with
concurrency limiting. It uses the attack's params_type to create parameters
from seed groups.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `max_concurrency` | `int` | Maximum number of concurrent attack executions (default: 1). Defaults to `1`. |

**Methods:**

#### execute_attack_async

```python
execute_attack_async(attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT], objectives: Sequence[str], field_overrides: Optional[Sequence[dict[str, Any]]] = None, return_partial_on_failure: bool = False, broadcast_fields: Any = {}) → AttackExecutorResult[AttackStrategyResultT]
```

Execute attacks in parallel for each objective.

Creates AttackParameters directly from objectives and field values.

| Parameter | Type | Description |
|---|---|---|
| `attack` | `AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]` | The attack strategy to execute. |
| `objectives` | `Sequence[str]` | List of attack objectives. |
| `field_overrides` | `Optional[Sequence[dict[str, Any]]]` | Optional per-objective field overrides. If provided, must match the length of objectives. Defaults to `None`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results when some objectives fail. If False (default), raises the first exception. Defaults to `False`. |
| `**broadcast_fields` | `Any` | Fields applied to all objectives (e.g., memory_labels). Per-objective field_overrides take precedence. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackStrategyResultT]` — AttackExecutorResult with completed results and any incomplete objectives.

**Raises:**

- `ValueError` — If objectives is empty or field_overrides length doesn't match.
- `BaseException` — If return_partial_on_failure=False and any objective fails.

#### execute_attack_from_seed_groups_async

```python
execute_attack_from_seed_groups_async(attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT], seed_groups: Sequence[SeedAttackGroup], adversarial_chat: Optional[PromptChatTarget] = None, objective_scorer: Optional[TrueFalseScorer] = None, field_overrides: Optional[Sequence[dict[str, Any]]] = None, return_partial_on_failure: bool = False, broadcast_fields: Any = {}) → AttackExecutorResult[AttackStrategyResultT]
```

Execute attacks in parallel, extracting parameters from SeedAttackGroups.

Uses the attack's params_type.from_seed_group() to extract parameters,
automatically handling which fields the attack accepts.

| Parameter | Type | Description |
|---|---|---|
| `attack` | `AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]` | The attack strategy to execute. |
| `seed_groups` | `Sequence[SeedAttackGroup]` | SeedAttackGroups containing objectives and optional prompts. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | Optional chat target for generating adversarial prompts or simulated conversations. Required when seed groups contain SeedSimulatedConversation configurations. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Optional scorer for evaluating simulated conversations. Required when seed groups contain SeedSimulatedConversation configurations. Defaults to `None`. |
| `field_overrides` | `Optional[Sequence[dict[str, Any]]]` | Optional per-seed-group field overrides. If provided, must match the length of seed_groups. Each dict is passed to from_seed_group() as overrides. Defaults to `None`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results when some objectives fail. If False (default), raises the first exception. Defaults to `False`. |
| `**broadcast_fields` | `Any` | Fields applied to all seed groups (e.g., memory_labels). Per-seed-group field_overrides take precedence. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackStrategyResultT]` — AttackExecutorResult with completed results and any incomplete objectives.

**Raises:**

- `ValueError` — If seed_groups is empty or field_overrides length doesn't match.
- `BaseException` — If return_partial_on_failure=False and any objective fails.

#### execute_multi_objective_attack_async

```python
execute_multi_objective_attack_async(attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT], objectives: list[str], prepended_conversation: Optional[list[Message]] = None, memory_labels: Optional[dict[str, str]] = None, return_partial_on_failure: bool = False, attack_params: Any = {}) → AttackExecutorResult[AttackStrategyResultT]
```

Execute the same attack strategy with multiple objectives against the same target in parallel.

.. deprecated::
    Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

| Parameter | Type | Description |
|---|---|---|
| `attack` | `AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]` | The attack strategy to use for all objectives. |
| `objectives` | `list[str]` | List of attack objectives to test. |
| `prepended_conversation` | `Optional[list[Message]]` | Conversation to prepend to the target model. Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Additional labels that can be applied to the prompts. Defaults to `None`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results on failure. Defaults to `False`. |
| `**attack_params` | `Any` | Additional parameters specific to the attack strategy. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackStrategyResultT]` — AttackExecutorResult with completed results and any incomplete objectives.

#### execute_multi_turn_attacks_async

```python
execute_multi_turn_attacks_async(attack: AttackStrategy[_MultiTurnContextT, AttackStrategyResultT], objectives: list[str], messages: Optional[list[Message]] = None, prepended_conversations: Optional[list[list[Message]]] = None, memory_labels: Optional[dict[str, str]] = None, return_partial_on_failure: bool = False, attack_params: Any = {}) → AttackExecutorResult[AttackStrategyResultT]
```

Execute a batch of multi-turn attacks with multiple objectives.

.. deprecated::
    Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

| Parameter | Type | Description |
|---|---|---|
| `attack` | `AttackStrategy[_MultiTurnContextT, AttackStrategyResultT]` | The multi-turn attack strategy to use. |
| `objectives` | `list[str]` | List of attack objectives to test. |
| `messages` | `Optional[list[Message]]` | List of messages to use for this execution (per-objective). Defaults to `None`. |
| `prepended_conversations` | `Optional[list[list[Message]]]` | Conversations to prepend to each objective (per-objective). Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Additional labels that can be applied to the prompts. Defaults to `None`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results on failure. Defaults to `False`. |
| `**attack_params` | `Any` | Additional parameters specific to the attack strategy. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackStrategyResultT]` — AttackExecutorResult with completed results and any incomplete objectives.

**Raises:**

- `TypeError` — If the attack does not use MultiTurnAttackContext.

#### execute_single_turn_attacks_async

```python
execute_single_turn_attacks_async(attack: AttackStrategy[_SingleTurnContextT, AttackStrategyResultT], objectives: list[str], messages: Optional[list[Message]] = None, prepended_conversations: Optional[list[list[Message]]] = None, memory_labels: Optional[dict[str, str]] = None, return_partial_on_failure: bool = False, attack_params: Any = {}) → AttackExecutorResult[AttackStrategyResultT]
```

Execute a batch of single-turn attacks with multiple objectives.

.. deprecated::
    Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

| Parameter | Type | Description |
|---|---|---|
| `attack` | `AttackStrategy[_SingleTurnContextT, AttackStrategyResultT]` | The single-turn attack strategy to use. |
| `objectives` | `list[str]` | List of attack objectives to test. |
| `messages` | `Optional[list[Message]]` | List of messages to use for this execution (per-objective). Defaults to `None`. |
| `prepended_conversations` | `Optional[list[list[Message]]]` | Conversations to prepend to each objective (per-objective). Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Additional labels that can be applied to the prompts. Defaults to `None`. |
| `return_partial_on_failure` | `bool` | If True, returns partial results on failure. Defaults to `False`. |
| `**attack_params` | `Any` | Additional parameters specific to the attack strategy. Defaults to `{}`. |

**Returns:**

- `AttackExecutorResult[AttackStrategyResultT]` — AttackExecutorResult with completed results and any incomplete objectives.

**Raises:**

- `TypeError` — If the attack does not use SingleTurnAttackContext.

## `class AttackExecutorResult(Generic[AttackResultT])`

Result container for attack execution, supporting both full and partial completion.

This class holds results from parallel attack execution. It is iterable and
behaves like a list in the common case where all objectives complete successfully.

When some objectives don't complete (throw exceptions), access incomplete_objectives
to retrieve the failures, or use raise_if_incomplete() to raise the first exception.

Note: "completed" means the execution finished, not that the attack objective was achieved.

**Methods:**

#### `get_results() → list[AttackResultT]`

Get completed results, raising if any incomplete.

**Returns:**

- `list[AttackResultT]` — List of completed attack results.

#### `raise_if_incomplete() → None`

Raise the first exception if any objectives are incomplete.

## `class AttackParameters`

Immutable parameters for attack execution.

This class defines the standard contract for attack parameters. All attacks
at a given level of the hierarchy share the same parameter signature.

Attacks that don't accept certain parameters should use the `excluding()` factory
to create a derived params type without those fields. Attacks that need additional
parameters should extend this class with new fields.

**Methods:**

#### `excluding(field_names: str = ()) → type[AttackParameters]`

Create a new AttackParameters subclass that excludes the specified fields.

This factory method creates a frozen dataclass without the specified fields.
The resulting class inherits the `from_seed_group()` behavior and will raise
if excluded fields are passed as overrides.

| Parameter | Type | Description |
|---|---|---|
| `*field_names` | `str` | Names of fields to exclude from the new params type. Defaults to `()`. |

**Returns:**

- `type[AttackParameters]` — A new AttackParameters subclass without the specified fields.

**Raises:**

- `ValueError` — If any field_name is not a valid field of this class.

#### from_seed_group_async

```python
from_seed_group_async(seed_group: SeedAttackGroup, adversarial_chat: Optional[PromptChatTarget] = None, objective_scorer: Optional[TrueFalseScorer] = None, overrides: Any = {}) → AttackParamsT
```

Create an AttackParameters instance from a SeedAttackGroup.

Extracts standard fields from the seed group and applies any overrides.
If the seed_group has a simulated conversation config,
generates the simulated conversation using the provided adversarial_chat and scorer.

| Parameter | Type | Description |
|---|---|---|
| `seed_group` | `SeedAttackGroup` | The seed attack group to extract parameters from. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | The adversarial chat target for generating simulated conversations. Required if seed_group has a simulated conversation config. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | The scorer for evaluating simulated conversations. Required if seed_group has a simulated conversation config. Defaults to `None`. |
| `**overrides` | `Any` | Field overrides to apply. Must be valid fields for this params type. Defaults to `{}`. |

**Returns:**

- `AttackParamsT` — An instance of this AttackParameters type.

**Raises:**

- `ValueError` — If seed_group has no objective or if overrides contain invalid fields.
- `ValueError` — If seed_group has simulated conversation but adversarial_chat/scorer not provided.

## `class AttackScoringConfig`

Scoring configuration for evaluating attack effectiveness.

This class defines the scoring components used to evaluate attack effectiveness,
detect refusals, and perform auxiliary scoring operations.

## `class AttackStrategy(Strategy[AttackStrategyContextT, AttackStrategyResultT], Identifiable, ABC)`

Abstract base class for attack strategies.
Defines the interface for executing attacks and handling results.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. |
| `context_type` | `type[AttackStrategyContextT]` | The type of context this strategy operates on. |
| `params_type` | `Type[AttackParamsT]` | The type of parameters this strategy accepts. Defaults to AttackParameters. Use AttackParameters.excluding() to create a params type that rejects certain fields. Defaults to `AttackParameters`. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → AttackStrategyResultT`

Execute the attack strategy asynchronously with the provided parameters.

This method provides a stable contract for all attacks. The signature includes
all standard parameters (objective, next_message, prepended_conversation, memory_labels).
Attacks that don't accept certain parameters will raise ValueError if those
parameters are provided.

| Parameter | Type | Description |
|---|---|---|
| `objective` | `str` | The objective of the attack. |
| `next_message` | `Optional[Message]` | Message to send to the target. |
| `prepended_conversation` | `Optional[List[Message]]` | Conversation to prepend. |
| `memory_labels` | `Optional[Dict[str, str]]` | Memory labels for the attack context. |
| `**kwargs` | `Any` | Additional context-specific parameters (conversation_id, system_prompt, etc.). Defaults to `{}`. |

**Returns:**

- `AttackStrategyResultT` — The result of the attack execution.

**Raises:**

- `ValueError` — If required parameters are missing or if unsupported parameters are provided.

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration, or None if not applicable.

#### `get_objective_target() → PromptTarget`

Get the objective target for this attack strategy.

**Returns:**

- `PromptTarget` — The target system being attacked.

#### `get_request_converters() → list[Any]`

Get request converter configurations used by this strategy.

**Returns:**

- `list[Any]` — list[Any]: The list of request PromptConverterConfiguration objects.
