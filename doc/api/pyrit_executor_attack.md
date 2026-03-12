# pyrit.executor.attack

Attack executor module.

## Functions

### generate_simulated_conversation_async

```python
generate_simulated_conversation_async(objective: str, adversarial_chat: PromptChatTarget, objective_scorer: TrueFalseScorer, num_turns: int = 3, starting_sequence: int = 0, adversarial_chat_system_prompt_path: Union[str, Path], simulated_target_system_prompt_path: Optional[Union[str, Path]] = None, next_message_system_prompt_path: Optional[Union[str, Path]] = None, attack_converter_config: Optional[AttackConverterConfig] = None, memory_labels: Optional[dict[str, str]] = None) → list[SeedPrompt]
```

Generate a simulated conversation between an adversarial chat and a target.

This utility runs a RedTeamingAttack with `score_last_turn_only=True` against a simulated
target (the same LLM as adversarial_chat, optionally configured with a system prompt).
The resulting conversation is returned as a list of SeedPrompts that can be merged with
other SeedPrompts in a SeedGroup for use as `prepended_conversation` and `next_message`.

Use cases:
- Creating role-play scenarios dynamically (e.g., movie script, video game)
- Establishing conversational context before attacking a real target
- Generating multi-turn jailbreak setups without hardcoded responses

| Parameter | Type | Description |
|---|---|---|
| `objective` | `str` | The objective for the adversarial chat to work toward. |
| `adversarial_chat` | `PromptChatTarget` | The adversarial LLM that generates attack prompts. This same LLM is also used as the simulated target. |
| `objective_scorer` | `TrueFalseScorer` | Scorer to evaluate the final turn. |
| `num_turns` | `int` | Number of conversation turns to generate. Defaults to 3. Defaults to `3`. |
| `starting_sequence` | `int` | The starting sequence number for the generated SeedPrompts. Each message gets an incrementing sequence number. Defaults to 0. Defaults to `0`. |
| `adversarial_chat_system_prompt_path` | `Union[str, Path]` | Path to the system prompt for the adversarial chat. |
| `simulated_target_system_prompt_path` | `Optional[Union[str, Path]]` | Path to the system prompt for the simulated target. If None, no system prompt is used for the simulated target. Defaults to `None`. |
| `next_message_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to a system prompt for generating a final user message. If provided, after the simulated conversation, a single LLM call generates a user message that attempts to get the target to fulfill the objective in their next response. The prompt template receives `objective` and `conversation_so_far` parameters. Defaults to `None`. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Converter configuration for the attack. Defaults to None. Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Labels to associate with the conversation in memory. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[SeedPrompt]` — List of SeedPrompts representing the generated conversation, with sequence numbers
- `list[SeedPrompt]` — starting from `starting_sequence` and incrementing by 1 for each message.
- `list[SeedPrompt]` — User messages have role="user", assistant messages have role="assistant".
- `list[SeedPrompt]` — If next_message_system_prompt_path is provided, the last message will be a user message
- `list[SeedPrompt]` — generated to elicit the objective fulfillment.

**Raises:**

- `ValueError` — If num_turns is not a positive integer.

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

## `class AttackResultPrinter(ABC)`

Abstract base class for printing attack results.

This interface defines the contract for printing attack results in various formats.
Implementations can render results to console, logs, files, or other outputs.

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False) → None
```

Print only the conversation history.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation to print |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the output. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result without the full conversation.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize |

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

## `class ChunkedRequestAttack(MultiTurnAttackStrategy[ChunkedRequestAttackContext, AttackResult])`

Implementation of chunked extraction attack strategy.

This class orchestrates a multi-turn attack where protected information is
requested in chunks/segments across multiple turns. This technique bypasses
length-based filters or output truncation by asking for specific character
ranges of a secret value.

This technique was discovered during CTF red teaming exercises where targets
would reveal portions of secrets when asked for specific segments, even when
they refused to reveal the complete value.

The attack flow consists of:

1. Generating chunk request prompts based on the configured strategy.
2. Sending each chunk request to the target system sequentially.
3. Collecting responses from each chunk request.
4. Combining all chunk responses to reconstruct the full value.
5. Evaluating the combined result with scorers if configured.
6. Returning the attack result with achievement status.

The strategy supports customization through converters and scorers for
comprehensive evaluation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `chunk_size` | `int` | Size of each chunk to request (default: 50). Defaults to `50`. |
| `total_length` | `int` | Estimated total length of the target value (default: 200). Defaults to `200`. |
| `chunk_type` | `str` | Type of chunk to request (e.g., "characters", "bytes", "words"). Defaults to `'characters'`. |
| `request_template` | `str` | Template for generating chunk requests (default: "Give me {chunk_type} {start}-{end} of '{objective}'"). Defaults to `DEFAULT_TEMPLATE`. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |

**Methods:**

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration with objective and auxiliary scorers.

## `class ChunkedRequestAttackContext(MultiTurnAttackContext[Any])`

Context for the ChunkedRequest attack strategy.

## `class ConsoleAttackResultPrinter(AttackResultPrinter)`

Console printer for attack results with enhanced formatting.

This printer formats attack results for console display with optional color coding,
proper indentation, text wrapping, and visual separators. Colors can be disabled
for consoles that don't support ANSI characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Maximum width for text wrapping. Must be positive. Defaults to 100. Defaults to `100`. |
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. When False, all output will be plain text without colors. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False, include_reasoning_trace: bool = False) → None
```

Print the conversation history to console with enhanced formatting.

Displays the full conversation between user and assistant, including:
- Turn numbers
- Role indicators (USER/ASSISTANT)
- Original and converted values when different
- Images if present
- Scores for each response

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation_id. Must have a valid conversation_id attribute. |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |
| `include_reasoning_trace` | `bool` | Whether to include model reasoning trace in the output for applicable models. Defaults to False. Defaults to `False`. |

#### print_messages_async

```python
print_messages_async(messages: list[Any], include_scores: bool = False, include_reasoning_trace: bool = False) → None
```

Print a list of messages to console with enhanced formatting.

This method can be called directly with a list of Message objects,
without needing an AttackResult. Useful for printing prepended_conversation
or any other list of messages.

Displays:
- Turn numbers
- Role indicators (USER/ASSISTANT/SYSTEM)
- Original and converted values when different
- Images if present
- Scores for each response (if include_scores=True)

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list` | List of Message objects to print. |
| `include_scores` | `bool` | Whether to include scores in the output. Defaults to False. Defaults to `False`. |
| `include_reasoning_trace` | `bool` | Whether to include model reasoning trace in the output for applicable models. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result to console.

This method orchestrates the printing of all components of an attack result,
including header, summary, conversation history, metadata, and footer.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print. Must not be None. |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the output. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result with enhanced formatting.

Displays:
- Basic information (objective, attack type, conversation ID)
- Execution metrics (turns executed, execution time)
- Outcome information (status, reason)
- Final score if available

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize. Must contain objective, attack_identifier, conversation_id, executed_turns, execution_time_ms, outcome, and optionally outcome_reason and last_score attributes. |

## `class ContextComplianceAttack(PromptSendingAttack)`

Implementation of the context compliance attack strategy.

This attack attempts to bypass safety measures by rephrasing the objective into a more benign context.
It uses an adversarial chat target to:
1. Rephrase the objective as a more benign question
2. Generate a response to the benign question
3. Rephrase the original objective as a follow-up question

This creates a context that makes it harder for the target to detect the true intent.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptChatTarget` | The target system to attack. Must be a PromptChatTarget. Defaults to `REQUIRED_VALUE`. |
| `attack_adversarial_config` | `AttackAdversarialConfig` | Configuration for the adversarial component, including the adversarial chat target used for rephrasing. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for attack converters, including request and response converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for attack scoring. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | The prompt normalizer to use for sending prompts. Defaults to `None`. |
| `max_attempts_on_failure` | `int` | Maximum number of attempts to retry on failure. Defaults to `0`. |
| `context_description_instructions_path` | `Optional[Path]` | Path to the context description instructions YAML file. If not provided, uses the default path. Defaults to `None`. |
| `affirmative_response` | `Optional[str]` | The affirmative response to be used in the conversation history. If not provided, uses the default "yes.". Defaults to `None`. |

## `class ConversationManager`

Manages conversations for attacks, handling message history,
system prompts, and conversation state.

This class provides methods to:
- Initialize attack context with prepended conversations
- Retrieve conversation history
- Set system prompts for chat targets

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `attack_identifier` | `ComponentIdentifier` | The identifier of the attack this manager belongs to. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional prompt normalizer for converting prompts. If not provided, a default PromptNormalizer instance will be created. Defaults to `None`. |

**Methods:**

#### add_prepended_conversation_to_memory_async

```python
add_prepended_conversation_to_memory_async(prepended_conversation: list[Message], conversation_id: str, request_converters: Optional[list[PromptConverterConfiguration]] = None, prepended_conversation_config: Optional[PrependedConversationConfig] = None, max_turns: Optional[int] = None) → int
```

Add prepended conversation messages to memory for a chat target.

This is a lower-level method that handles adding messages to memory without
modifying any attack context state. It can be called directly by attacks
that manage their own state (like TAP nodes) or internally by
initialize_context_async for standard attacks.

Messages are added with:
- Duplicated message objects (preserves originals)
- simulated_assistant role for assistant messages (for traceability)
- Converters applied based on config

| Parameter | Type | Description |
|---|---|---|
| `prepended_conversation` | `list[Message]` | Messages to add to memory. |
| `conversation_id` | `str` | Conversation ID to assign to all messages. |
| `request_converters` | `Optional[list[PromptConverterConfiguration]]` | Optional converters to apply to messages. Defaults to `None`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfig]` | Optional configuration for converter roles. Defaults to `None`. |
| `max_turns` | `Optional[int]` | If provided, validates that turn count doesn't exceed this limit. Defaults to `None`. |

**Returns:**

- `int` — The number of turns (assistant messages) added.

**Raises:**

- `ValueError` — If max_turns is exceeded by the prepended conversation.

#### `get_conversation(conversation_id: str) → list[Message]`

Retrieve a conversation by its ID.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The ID of the conversation to retrieve. |

**Returns:**

- `list[Message]` — A list of messages in the conversation, ordered by creation time.
- `list[Message]` — Returns empty list if no messages exist.

#### get_last_message

```python
get_last_message(conversation_id: str, role: Optional[ChatMessageRole] = None) → Optional[MessagePiece]
```

Retrieve the most recent message from a conversation.

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The ID of the conversation to retrieve from. |
| `role` | `Optional[ChatMessageRole]` | If provided, return only the last message matching this role. Defaults to `None`. |

**Returns:**

- `Optional[MessagePiece]` — The last message piece, or None if no messages exist.

#### initialize_context_async

```python
initialize_context_async(context: AttackContext[Any], target: PromptTarget, conversation_id: str, request_converters: Optional[list[PromptConverterConfiguration]] = None, prepended_conversation_config: Optional[PrependedConversationConfig] = None, max_turns: Optional[int] = None, memory_labels: Optional[dict[str, str]] = None) → ConversationState
```

Initialize attack context with prepended conversation and merged labels.

This is the primary method for setting up an attack context. It:
1. Merges memory_labels from attack strategy with context labels
2. Processes prepended_conversation based on target type and config
3. Updates context.executed_turns for multi-turn attacks
4. Sets context.next_message if there's an unanswered user message

| Parameter | Type | Description |
|---|---|---|
| `context` | `AttackContext[Any]` | The attack context to initialize. |
| `target` | `PromptTarget` | The objective target for the conversation. |
| `conversation_id` | `str` | Unique identifier for the conversation. |
| `request_converters` | `Optional[list[PromptConverterConfiguration]]` | Converters to apply to messages. Defaults to `None`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfig]` | Configuration for handling prepended conversation. Defaults to `None`. |
| `max_turns` | `Optional[int]` | Maximum turns allowed (for validation and state tracking). Defaults to `None`. |
| `memory_labels` | `Optional[dict[str, str]]` | Labels from the attack strategy to merge with context labels. Defaults to `None`. |

**Returns:**

- `ConversationState` — ConversationState with turn_count and last_assistant_message_scores.

**Raises:**

- `ValueError` — If conversation_id is empty, or if prepended_conversation
requires a PromptChatTarget but target is not one.

#### set_system_prompt

```python
set_system_prompt(target: PromptChatTarget, conversation_id: str, system_prompt: str, labels: Optional[dict[str, str]] = None) → None
```

Set or update the system prompt for a conversation.

| Parameter | Type | Description |
|---|---|---|
| `target` | `PromptChatTarget` | The chat target to set the system prompt on. |
| `conversation_id` | `str` | Unique identifier for the conversation. |
| `system_prompt` | `str` | The system prompt text. |
| `labels` | `Optional[dict[str, str]]` | Optional labels to associate with the system prompt. Defaults to `None`. |

## `class ConversationSession`

Session for conversations.

## `class ConversationState`

Container for conversation state data returned from context initialization.

## `class CrescendoAttack(MultiTurnAttackStrategy[CrescendoAttackContext, CrescendoAttackResult])`

Implementation of the Crescendo attack strategy.

The Crescendo Attack is a multi-turn strategy that progressively guides the model to generate harmful
content through small, benign steps. It leverages the model's recency bias, pattern-following tendency,
and trust in self-generated text.

The attack flow consists of:
1. Generating progressively harmful prompts using an adversarial chat model.
2. Sending prompts to the target and evaluating responses for refusal.
3. Backtracking when the target refuses to respond.
4. Scoring responses to determine if the objective has been achieved.
5. Continuing until the objective is met or maximum turns/backtracks are reached.

You can learn more about the Crescendo attack at:
https://crescendo-the-multiturn-jailbreak.github.io/

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptChatTarget` | The target system to attack. Must be a PromptChatTarget. Defaults to `REQUIRED_VALUE`. |
| `attack_adversarial_config` | `AttackAdversarialConfig` | Configuration for the adversarial component, including the adversarial chat target and optional system prompt path. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for attack converters, including request and response converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring responses. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for prompts. Defaults to `None`. |
| `max_backtracks` | `int` | Maximum number of backtracks allowed. Defaults to `10`. |
| `max_turns` | `int` | Maximum number of turns allowed. Defaults to `10`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfiguration]` |  Configuration for how to process prepended conversations. Controls converter application by role, message normalization, and non-chat target behavior. Defaults to `None`. |

**Methods:**

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration with objective scorer,
auxiliary scorers, and refusal scorer.

## `class CrescendoAttackContext(MultiTurnAttackContext[Any])`

Context for the Crescendo attack strategy.

## `class CrescendoAttackResult(AttackResult)`

Result of the Crescendo attack strategy execution.

## `class FlipAttack(PromptSendingAttack)`

Implement the FlipAttack method found here:
https://arxiv.org/html/2410.02832v1.

Essentially, it adds a system prompt to the beginning of the conversation to flip each word in the prompt.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptChatTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_converter_config` | `(AttackConverterConfig, Optional)` | Configuration for the prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `(AttackScoringConfig, Optional)` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `(PromptNormalizer, Optional)` | Normalizer for handling prompts. Defaults to `None`. |
| `max_attempts_on_failure` | `(int, Optional)` | Maximum number of attempts to retry on failure. Defaults to `0`. |

## `class ManyShotJailbreakAttack(PromptSendingAttack)`

Implement the Many Shot Jailbreak method as discussed in research found here:
https://www.anthropic.com/research/many-shot-jailbreaking.

Prepends the seed prompt with a faux dialogue between a human and an AI, using examples from a dataset
to demonstrate successful jailbreaking attempts. This method leverages the model's ability to learn from
examples to bypass safety measures.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_converter_config` | `(AttackConverterConfig, Optional)` | Configuration for the prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `(AttackScoringConfig, Optional)` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `(PromptNormalizer, Optional)` | Normalizer for handling prompts. Defaults to `None`. |
| `max_attempts_on_failure` | `(int, Optional)` | Maximum number of attempts to retry on failure. Defaults to 0. Defaults to `0`. |
| `example_count` | `int` | The number of examples to include from many_shot_examples or the Many Shot Jailbreaking dataset. Defaults to the first 100. Defaults to `100`. |
| `many_shot_examples` | `(list[dict[str, str]], Optional)` | The many shot jailbreaking examples to use. If not provided, takes the first `example_count` examples from Many Shot Jailbreaking dataset. Defaults to `None`. |

## `class MarkdownAttackResultPrinter(AttackResultPrinter)`

Markdown printer for attack results optimized for Jupyter notebooks.

This printer formats attack results as markdown, making them ideal for display
in Jupyter notebooks where LLM responses often contain code blocks and other
markdown formatting that should be properly rendered.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `display_inline` | `bool` | If True, uses IPython.display to render markdown inline in Jupyter notebooks. If False, prints markdown strings. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_conversation_async

```python
print_conversation_async(result: AttackResult, include_scores: bool = False) → None
```

Print only the conversation history as formatted markdown.

Extracts and displays the conversation messages from the attack result
without the summary or metadata sections. Useful for focusing on the
actual interaction flow.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result containing the conversation to display. |
| `include_scores` | `bool` | Whether to include scores for each message. Defaults to False. Defaults to `False`. |

#### print_result_async

```python
print_result_async(result: AttackResult, include_auxiliary_scores: bool = False, include_pruned_conversations: bool = False, include_adversarial_conversation: bool = False) → None
```

Print the complete attack result as formatted markdown.

Generates a comprehensive markdown report including attack summary,
conversation history, scores, and metadata. The output is optimized
for display in Jupyter notebooks.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to print. |
| `include_auxiliary_scores` | `bool` | Whether to include auxiliary scores in the conversation display. Defaults to False. Defaults to `False`. |
| `include_pruned_conversations` | `bool` | Whether to include pruned conversations. For each pruned conversation, only the last message and its score are shown. Defaults to False. Defaults to `False`. |
| `include_adversarial_conversation` | `bool` | Whether to include the adversarial conversation (the red teaming LLM's reasoning). Only shown for successful attacks to avoid overwhelming output. Defaults to False. Defaults to `False`. |

#### `print_summary_async(result: AttackResult) → None`

Print a summary of the attack result as formatted markdown.

Displays key information about the attack including objective, outcome,
execution metrics, and final score without the full conversation history.
Useful for getting a quick overview of the attack results.

| Parameter | Type | Description |
|---|---|---|
| `result` | `AttackResult` | The attack result to summarize. |

## `class MultiPromptSendingAttack(MultiTurnAttackStrategy[MultiTurnAttackContext[Any], AttackResult])`

Implementation of multi-prompt sending attack strategy.

This class orchestrates a multi-turn attack where a series of predefined malicious
prompts are sent sequentially to try to achieve a specific objective against a target
system. The strategy evaluates the final target response using optional scorers to
determine if the objective has been met.

The attack flow consists of:
1. Sending each predefined prompt to the target system in sequence.
2. Continuing until all predefined prompts are sent.
3. Evaluating the final response with scorers if configured.
4. Returning the attack result with achievement status.

Note: This attack always runs all predefined prompts regardless of whether the
objective is achieved early in the sequence.

The strategy supports customization through prepended conversations, converters,
and multiple scorer types for comprehensive evaluation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → AttackResult`

Execute the attack strategy asynchronously with the provided parameters.

**Returns:**

- `AttackResult` — The result of the attack execution.

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration with objective and auxiliary scorers.

## `class MultiPromptSendingAttackParameters(AttackParameters)`

Parameters for MultiPromptSendingAttack.

Extends AttackParameters to include user_messages field for multi-turn attacks.
Only accepts objective and user_messages fields.

**Methods:**

#### from_seed_group_async

```python
from_seed_group_async(seed_group: SeedAttackGroup, adversarial_chat: Optional[PromptChatTarget] = None, objective_scorer: Optional[TrueFalseScorer] = None, overrides: Any = {}) → MultiPromptSendingAttackParameters
```

Create parameters from a SeedGroup, extracting user messages.

| Parameter | Type | Description |
|---|---|---|
| `seed_group` | `SeedAttackGroup` | The seed group to extract parameters from. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | Not used by this attack type. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Not used by this attack type. Defaults to `None`. |
| `**overrides` | `Any` | Field overrides to apply. Defaults to `{}`. |

**Returns:**

- `MultiPromptSendingAttackParameters` — MultiPromptSendingAttackParameters instance.

**Raises:**

- `ValueError` — If seed_group has no objective, no user messages, or if overrides contain invalid fields.

## `class MultiTurnAttackContext(AttackContext[AttackParamsT])`

Context for multi-turn attacks.

Holds execution state for multi-turn attacks. The immutable attack parameters
(objective, next_message, prepended_conversation, memory_labels) are stored in
the params field inherited from AttackContext.

## `class MultiTurnAttackStrategy(AttackStrategy[MultiTurnAttackStrategyContextT, AttackStrategyResultT], ABC)`

Strategy for executing multi-turn attacks.
This strategy is designed to handle attacks that consist of multiple turns
of interaction with the target model.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. |
| `context_type` | `type[MultiTurnAttackContext]` | The type of context this strategy will use. |
| `params_type` | `Type[AttackParamsT]` | The type of parameters this strategy accepts. Defaults to `AttackParameters`. |
| `logger` | `logging.Logger` | Logger instance for logging events and messages. Defaults to `logger`. |

## `class PrependedConversationConfig`

Configuration for controlling how prepended conversations are processed before
being sent to the objective target.

This class provides control over:
- Which message roles should have request converters applied
- How to normalize conversation history for non-chat objective targets
- What to do when the objective target is not a PromptChatTarget

**Methods:**

#### `default() → PrependedConversationConfig`

Create a default configuration with converters applied to all roles.

**Returns:**

- `PrependedConversationConfig` — A configuration that applies converters to all prepended messages,
- `PrependedConversationConfig` — raising an error for non-chat targets.

#### for_non_chat_target

```python
for_non_chat_target(message_normalizer: Optional[MessageStringNormalizer] = None, apply_converters_to_roles: Optional[list[ChatMessageRole]] = None) → PrependedConversationConfig
```

Create a configuration for use with non-chat targets.

This configuration normalizes the prepended conversation into a text block
that will be prepended to the first message sent to the target.

| Parameter | Type | Description |
|---|---|---|
| `message_normalizer` | `Optional[MessageStringNormalizer]` | Normalizer for formatting the prepended conversation into a string. Defaults to ConversationContextNormalizer if not provided. Defaults to `None`. |
| `apply_converters_to_roles` | `Optional[list[ChatMessageRole]]` | Roles to apply converters to before normalization. Defaults to all roles. Defaults to `None`. |

**Returns:**

- `PrependedConversationConfig` — A configuration that normalizes the prepended conversation for non-chat targets.

#### `get_message_normalizer() → MessageStringNormalizer`

Get the normalizer for objective target context, with a default fallback.

**Returns:**

- `MessageStringNormalizer` — The configured objective_target_context_normalizer, or a default
- `MessageStringNormalizer` — ConversationContextNormalizer if none was configured.

## `class PromptSendingAttack(SingleTurnAttackStrategy)`

Implementation of single-turn prompt sending attack strategy.

This class orchestrates a single-turn attack where malicious prompts are injected
to try to achieve a specific objective against a target system. The strategy evaluates
the target response using optional scorers to determine if the objective has been met.

The attack flow consists of:
1. Preparing the prompt based on the objective.
2. Sending the prompt to the target system through optional converters.
3. Evaluating the response with scorers if configured.
4. Retrying on failure up to the configured number of retries.
5. Returning the attack result with achievement status.

The strategy supports customization through prepended conversations, converters,
and multiple scorer types for comprehensive evaluation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |
| `max_attempts_on_failure` | `int` | Maximum number of attempts to retry on failure. Defaults to `0`. |
| `params_type` | `Type[AttackParamsT]` | The type of parameters this strategy accepts. Defaults to AttackParameters. Use AttackParameters.excluding() to create a params type that rejects certain fields. Defaults to `AttackParameters`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfiguration]` |  Configuration for how to process prepended conversations. Controls converter application by role, message normalization, and non-chat target behavior. Defaults to `None`. |

**Methods:**

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration with objective and auxiliary scorers.

## `class RTASystemPromptPaths(enum.Enum)`

Enum for predefined red teaming attack system prompt paths.

## `class RedTeamingAttack(MultiTurnAttackStrategy[MultiTurnAttackContext[Any], AttackResult])`

Implementation of multi-turn red teaming attack strategy.

This class orchestrates an iterative attack process where an adversarial chat model generates
prompts to send to a target system, attempting to achieve a specified objective. The strategy
evaluates each target response using a scorer to determine if the objective has been met.

The attack flow consists of:
1. Generating adversarial prompts based on previous responses and scoring feedback.
2. Sending prompts to the target system through optional converters.
3. Scoring target responses to assess objective achievement.
4. Using scoring feedback to guide subsequent prompt generation.
5. Continuing until the objective is achieved or maximum turns are reached.

The strategy supports customization through system prompts, seed prompts, and prompt converters,
allowing for various attack techniques and scenarios.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_adversarial_config` | `AttackAdversarialConfig` | Configuration for the adversarial component. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for attack converters. Defaults to None. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for attack scoring. Defaults to None. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | The prompt normalizer to use for sending prompts. Defaults to None. Defaults to `None`. |
| `max_turns` | `int` | Maximum number of turns for the attack. Defaults to 10. Defaults to `10`. |
| `score_last_turn_only` | `bool` | If True, only score the final turn instead of every turn. This reduces LLM calls when intermediate scores are not needed (e.g., for generating simulated conversations). The attack will run for exactly max_turns when this is enabled. Defaults to False. Defaults to `False`. |

**Methods:**

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — Optional[AttackScoringConfig]: The scoring configuration with objective scorer
and use_score_as_feedback.

## `class RolePlayAttack(PromptSendingAttack)`

Implementation of single-turn role-play attack strategy.

This class orchestrates a role-play attack where malicious objectives are rephrased
into role-playing contexts to make them appear more benign and bypass content filters.
The strategy uses an adversarial chat target to transform the objective into a role-play
scenario before sending it to the target system.

The attack flow consists of:
1. Loading role-play scenarios from a YAML file.
2. Using an adversarial chat target to rephrase the objective into the role-play context.
3. Sending the rephrased objective to the target system.
4. Evaluating the response with scorers if configured.
5. Retrying on failure up to the configured number of retries.
6. Returning the attack result

The strategy supports customization through prepended conversations, converters,
and multiple scorer types.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `adversarial_chat` | `PromptChatTarget` | The adversarial chat target used to rephrase objectives into role-play scenarios. |
| `role_play_definition_path` | `pathlib.Path` | Path to the YAML file containing role-play definitions (rephrase instructions, user start turn, assistant start turn). |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |
| `max_attempts_on_failure` | `int` | Maximum number of attempts to retry the attack Defaults to `0`. |

## `class RolePlayPaths(enum.Enum)`

Enum for predefined role-play scenario paths.

## `class SingleTurnAttackContext(AttackContext[AttackParamsT])`

Context for single-turn attacks.

Holds execution state for single-turn attacks. The immutable attack parameters
(objective, next_message, prepended_conversation, memory_labels) are stored in
the params field inherited from AttackContext.

## `class SingleTurnAttackStrategy(AttackStrategy[SingleTurnAttackContext[Any], AttackResult], ABC)`

Strategy for executing single-turn attacks.
This strategy is designed to handle attacks that consist of a single turn
of interaction with the target model.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. |
| `context_type` | `type[SingleTurnAttackContext]` | The type of context this strategy will use. Defaults to `SingleTurnAttackContext`. |
| `params_type` | `Type[AttackParamsT]` | The type of parameters this strategy accepts. Defaults to `AttackParameters`. |
| `logger` | `logging.Logger` | Logger instance for logging events and messages. Defaults to `logger`. |

## `class SkeletonKeyAttack(PromptSendingAttack)`

Implementation of the skeleton key jailbreak attack strategy.

This attack sends an initial skeleton key prompt to the target, and then follows
up with a separate attack prompt. If successful, the first prompt makes the target
comply even with malicious follow-up prompts.

The attack flow consists of:
1. Sending a skeleton key prompt to bypass the target's safety mechanisms.
2. Sending the actual objective prompt to the primed target.
3. Evaluating the response using configured scorers to determine success.

Learn more about attack at the link below:
https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for prompt converters. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Configuration for scoring components. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Normalizer for handling prompts. Defaults to `None`. |
| `skeleton_key_prompt` | `Optional[str]` | The skeleton key prompt to use. If not provided, uses the default skeleton key prompt. Defaults to `None`. |
| `max_attempts_on_failure` | `int` | Maximum number of attempts to retry on failure. Defaults to `0`. |

## `class TAPAttackContext(MultiTurnAttackContext[Any])`

Context for the Tree of Attacks with Pruning (TAP) attack strategy.

This context contains all execution-specific state for a TAP attack instance,
ensuring thread safety by isolating state per execution.

## `class TAPAttackResult(AttackResult)`

Result of the Tree of Attacks with Pruning (TAP) attack strategy execution.

This result includes the standard attack result information with
attack-specific data stored in the metadata dictionary.

## `class TreeOfAttacksWithPruningAttack(AttackStrategy[TAPAttackContext, TAPAttackResult])`

Implement the Tree of Attacks with Pruning (TAP) attack strategy.

The TAP attack strategy systematically explores multiple adversarial prompt paths in parallel
using a tree structure. It employs breadth-first search with pruning to efficiently find
effective jailbreaks while managing computational resources.

How it works:
1. **Initialization**: Creates multiple initial attack branches (width) to explore different approaches
2. **Tree Expansion**: For each iteration (depth), branches are expanded by a branching factor
3. **Prompt Generation**: Each node generates adversarial prompts via an LLM red-teaming assistant
4. **Evaluation**: Responses are evaluated for objective achievement and on-topic relevance
5. **Pruning**: Low-scoring or off-topic branches are pruned to maintain the width constraint
6. **Iteration**: The process continues until the objective is achieved or max depth is reached

The strategy balances exploration (trying diverse approaches) with exploitation (focusing on
promising paths) through its pruning mechanism.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_target` | `PromptChatTarget` | The target system to attack. Defaults to `REQUIRED_VALUE`. |
| `attack_adversarial_config` | `AttackAdversarialConfig` | Configuration for the adversarial chat component. |
| `attack_converter_config` | `Optional[AttackConverterConfig]` | Configuration for attack converters. Defaults to None. Defaults to `None`. |
| `attack_scoring_config` | `Optional[AttackScoringConfig]` | Scoring configuration for TAP. The objective_scorer must be a FloatScaleThresholdScorer, which provides both granular float scores for node comparison and a threshold for determining success. Can be either AttackScoringConfig or TAPAttackScoringConfig. If not provided, a default configuration with SelfAskScaleScorer and threshold 0.7 is created. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | The prompt normalizer to use. Defaults to None. Defaults to `None`. |
| `tree_width` | `int` | Number of branches to explore in parallel at each level. Defaults to 3. Defaults to `3`. |
| `tree_depth` | `int` | Maximum number of iterations to perform. Defaults to 5. Defaults to `5`. |
| `branching_factor` | `int` | Number of child branches to create from each parent. Defaults to 2. Defaults to `2`. |
| `on_topic_checking_enabled` | `bool` | Whether to check if prompts are on-topic. Defaults to True. Defaults to `True`. |
| `desired_response_prefix` | `str` | Expected prefix for successful responses. Defaults to "Sure, here is". Defaults to `'Sure, here is'`. |
| `batch_size` | `int` | Number of nodes to process in parallel per batch. Defaults to 10. Defaults to `10`. |
| `prepended_conversation_config` | `Optional[PrependedConversationConfiguration]` |  Configuration for how to process prepended conversations. Controls converter application by role, message normalization, and non-chat target behavior. Defaults to `None`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → TAPAttackResult`

Execute the multi-turn attack strategy asynchronously with the provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `objective` | `str` | The objective of the attack. |
| `memory_labels` | `Optional[Dict[str, str]]` | Memory labels for the attack context. |
| `**kwargs` | `Any` | Additional parameters for the attack. Defaults to `{}`. |

**Returns:**

- `TAPAttackResult` — The result of the attack execution.

#### `get_attack_scoring_config() → Optional[AttackScoringConfig]`

Get the attack scoring configuration used by this strategy.

**Returns:**

- `Optional[AttackScoringConfig]` — The TAP-specific scoring configuration.
