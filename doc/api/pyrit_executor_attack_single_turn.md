# pyrit.executor.attack.single_turn

Singe turn attack strategies module.

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
