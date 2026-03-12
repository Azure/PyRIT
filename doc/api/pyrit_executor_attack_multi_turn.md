# pyrit.executor.attack.multi_turn

Multi-turn attack strategies module.

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

## `class ConversationSession`

Session for conversations.

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
