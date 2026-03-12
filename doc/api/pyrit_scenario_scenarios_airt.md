# pyrit.scenario.scenarios.airt

AIRT scenario classes.

## `class ContentHarms(Scenario)`

Content Harms Scenario implementation for PyRIT.

This scenario contains various harm-based checks that you can run to get a quick idea about model behavior
with respect to certain harm categories.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat` | `Optional[PromptChatTarget]` | Additionally used for scoring defaults. If not provided, a default OpenAI target will be created using environment variables. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Scorer to evaluate attack success. If not provided, creates a default composite scorer using Azure Content Filter and SelfAsk Refusal scorers. seed_dataset_prefix (Optional[str]): Prefix of the dataset to use to retrieve the objectives. This will be used to retrieve the appropriate seed groups from CentralMemory. If not provided, defaults to "content_harm". Defaults to `None`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |
| `objectives_by_harm` | `Optional[Dict[str, Sequence[SeedGroup]]]` | DEPRECATED - Use dataset_config in initialize_async instead. A dictionary mapping harm strategies to their corresponding SeedGroups. If not provided, default seed groups will be loaded from datasets. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with all content harm datasets.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — ContentHarmsStrategy.ALL

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The ContentHarmsStrategy enum class.

## `class ContentHarmsStrategy(ScenarioStrategy)`

ContentHarmsStrategy defines a set of strategies for testing model behavior
across several different harm categories. The scenario is designed to provide quick
feedback on model performance with respect to common harm types with the idea being that
users will dive deeper into specific harm categories based on initial results.

Each tag represents a different harm category that the model can be tested for.
Specifying the all tag will include a comprehensive test suite covering all harm categories.
Users can define objectives for each harm category via seed datasets or use the default datasets
provided with PyRIT.

## `class Cyber(Scenario)`

Cyber scenario implementation for PyRIT.

This scenario tests how willing models are to exploit cybersecurity harms by generating
malware. The Cyber class contains different variations of the malware generation
techniques.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat` | `Optional[PromptChatTarget]` | Adversarial chat for the red teaming attack, corresponding to CyberStrategy.MultiTurn. If not provided, defaults to an OpenAI chat target. Defaults to `None`. |
| `objectives` | `Optional[List[str]]` | Deprecated. Use dataset_config in initialize_async instead. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Objective scorer for malware detection. If not provided, defaults to a SelfAskScorer using the malware.yaml file under the scorer config store for malware detection Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. When True, a "baseline" attack is automatically added as the first atomic attack, allowing comparison between unmodified prompts and attack-modified prompts. Defaults to `True`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with airt_malware dataset.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — CyberStrategy.ALL (all cyber strategies).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The CyberStrategy enum class.

## `class CyberStrategy(ScenarioStrategy)`

Strategies for malware-focused cyber attacks. While not in the CyberStrategy class, a
few of these include:
* Shell smashing
* Zip bombs
* File deletion (rm -rf /).

## `class Jailbreak(Scenario)`

Jailbreak scenario implementation for PyRIT.

This scenario tests how vulnerable models are to jailbreak attacks by applying
various single-turn jailbreak templates to a set of test prompts. The responses are
scored to determine if the jailbreak was successful.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objective_scorer` | `Optional[TrueFalseScorer]` | Scorer for detecting successful jailbreaks (non-refusal). If not provided, defaults to an inverted refusal scorer. Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. Defaults to `False`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |
| `num_templates` | `Optional[int]` | Choose num_templates random jailbreaks rather than using all of them. Defaults to `None`. |
| `num_attempts` | `Optional[int]` | Number of times to try each jailbreak. Defaults to `1`. |
| `jailbreak_names` | `Optional[List[str]]` | List of jailbreak names from the template list under datasets. to use. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with airt_harms dataset.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — JailbreakStrategy.PromptSending.

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — type[ScenarioStrategy]: The JailbreakStrategy enum class.

#### `required_datasets() → list[str]`

Return a list of dataset names required by this scenario.

## `class JailbreakStrategy(ScenarioStrategy)`

Strategy for jailbreak attacks.

The SIMPLE strategy just sends the jailbroken prompt and records the response. It is meant to
expose an obvious way of using this scenario without worrying about additional tweaks and changes
to the prompt.

COMPLEX strategies use additional techniques to enhance the jailbreak like modifying the
system prompt or probing the target model for an additional vulnerability (e.g. the SkeletonKeyAttack).
They are meant to provide a sense of how well a jailbreak generalizes to slight changes in the delivery
method.

**Methods:**

#### `get_aggregate_tags() → set[str]`

Get the set of tags that represent aggregate categories.

**Returns:**

- `set[str]` — set[str]: Set of tags that are aggregate markers.

## `class Leakage(Scenario)`

Leakage scenario implementation for PyRIT.

This scenario tests how susceptible models are to leaking training data, PII, intellectual
property, or other confidential information. The LeakageScenario class contains different
attack variations designed to extract sensitive information from models.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat` | `Optional[PromptChatTarget]` | Adversarial chat target for multi-turn attacks (Crescendo, RolePlay). If not provided, defaults to an OpenAI chat target. Defaults to `None`. |
| `objectives` | `Optional[List[str]]` | List of objectives to test for data leakage. If not provided, defaults to objectives from the airt_leakage dataset. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Objective scorer for leakage detection. If not provided, defaults to a scorer that detects PII and training data leakage. Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. Defaults to `True`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with airt_leakage dataset.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — LeakageStrategy.ALL (all leakage strategies).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The LeakageStrategy enum class.

#### `required_datasets() → list[str]`

Return a list of dataset names required by this scenario.

## `class LeakageScenario(Leakage)`

Deprecated alias for Leakage.

This class is deprecated and will be removed in version 0.13.0.
Use `Leakage` instead.

## `class LeakageStrategy(ScenarioStrategy)`

Strategies for data leakage and privacy violation attacks.

These strategies test whether models can be manipulated into leaking training data,
intellectual property (IP), credentials, or other confidential content.

Leakage categories:
- ip: Intellectual property (copyrighted content, training data memorization)
- sensitive_data: Credentials, secrets, system prompts, API keys

**Methods:**

#### `get_aggregate_tags() → set[str]`

Get the set of tags that represent aggregate categories.

**Returns:**

- `set[str]` — Set[str]: Set of tags that represent aggregates, including "all",
     "single_turn", "multi_turn", "ip", and "sensitive_data".

## `class Psychosocial(Scenario)`

Psychosocial Harms Scenario implementation for PyRIT.

This scenario contains various psychosocial harm-based checks that evaluate model behavior
for responses that may harm the well-being or mental health of users.
Examples include mistreating users in a mental health crisis or aiding them in self-harm.

All datasets in this scenario share the unified 'psychosocial' harm category,
but each dataset (e.g., imminent_crisis, dependency) can have unique:
- Scoring rubrics
- Seed objectives

The scenario tests whether models:
- Recognize mental health crises and respond appropriately
- Avoid facilitating self-harm or harmful behaviors
- Provide crisis resources and encourage professional help

Example usage with custom configurations:

    # Define custom configurations per subharm category
    custom_configs = {
        "airt_imminent_crisis": SubharmConfig(
            crescendo_system_prompt_path="path/to/custom_escalation.yaml",
            scoring_rubric_path="path/to/custom_rubric.yaml",
        ),
    }

    scenario = Psychosocial(subharm_configs=custom_configs)
    await scenario.initialize_async(
        objective_target=target_llm,
        scenario_strategies=[PsychosocialStrategy.ImminentCrisis],
    )

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objectives` | `Optional[List[str]]` | DEPRECATED - Use dataset_config in initialize_async instead. List of objectives to test for psychosocial harms. Defaults to `None`. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | Additionally used for adversarial attacks and scoring defaults. If not provided, a default OpenAI target will be created using environment variables. Defaults to `None`. |
| `objective_scorer` | `Optional[FloatScaleThresholdScorer]` | Scorer to evaluate attack success. If not provided, creates a default scorer for evaluating psychosocial harms (crisis management, harmful facilitation, etc.). Defaults to `None`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |
| `subharm_configs` | `Optional[Dict[str, SubharmConfig]]` | Dictionary mapping subharm names to their SubharmConfig. Each seed in the dataset specifies its subharm in harm_categories[0], which is used to look up the appropriate config. Example: {     "airt_imminent_crisis": SubharmConfig(         crescendo_system_prompt_path="path/to/crisis_escalation.yaml",         scoring_rubric_path="path/to/crisis_management.yaml"     ),     "dependency": SubharmConfig(         crescendo_system_prompt_path="path/to/dependency_escalation.yaml",         scoring_rubric_path="path/to/dependency_rubric.yaml"     ), } If a subharm is not in this dict, falls back to defaults. Defaults to `None`. |
| `max_turns` | `int` | Maximum number of conversation turns for multi-turn attacks (CrescendoAttack). Defaults to 5. Increase for more gradual escalation, decrease for faster testing. Defaults to `5`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with psychosocial harm datasets.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — PsychosocialStrategy.ALL

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The PsychosocialHarmsStrategy enum class.

## `class PsychosocialScenario(Psychosocial)`

Deprecated alias for Psychosocial.

This class is deprecated and will be removed in version 0.13.0.
Use `Psychosocial` instead.

## `class PsychosocialStrategy(ScenarioStrategy)`

PsychosocialHarmsStrategy defines a set of strategies for testing model behavior
in psychosocial harm scenarios. The scenario is designed to evaluate how models handle
users in mental health crisis or if the model misrepresents itself as a licensed therapist.

The tags correspond to different attack strategies:
- single_turn: PromptSendingAttack and RolePlayAttack
- multi_turn: CrescendoAttack
- all: Both single_turn and multi_turn attacks

Specific strategies (imminent_crisis, licensed_therapist) filter seeds by harm_category.

## `class Scam(Scenario)`

Scam scenario evaluates an endpoint's ability to generate scam-related materials
(e.g., phishing emails, fraudulent messages) with primarily persuasion-oriented techniques.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `objectives` | `Optional[List[str]]` | List of objectives to test for scam-related harms. Defaults to `None`. |
| `objective_scorer` | `Optional[TrueFalseScorer]` | Custom scorer for objective evaluation. Defaults to `None`. |
| `adversarial_chat` | `Optional[PromptChatTarget]` | Chat target used to rephrase the objective into the role-play context (in single-turn strategies). Defaults to `None`. |
| `include_baseline` | `bool` | Whether to include a baseline atomic attack that sends all objectives without modifications. Defaults to True. When True, a "baseline" attack is automatically added as the first atomic attack, allowing comparison between unmodified prompts and encoding-modified prompts. Defaults to `True`. |
| `scenario_result_id` | `Optional[str]` | Optional ID of an existing scenario result to resume. Defaults to `None`. |

**Methods:**

#### `default_dataset_config() → DatasetConfiguration`

Return the default dataset configuration for this scenario.

**Returns:**

- `DatasetConfiguration` — Configuration with airt_scams dataset.

#### `get_default_strategy() → ScenarioStrategy`

Get the default strategy used when no strategies are specified.

**Returns:**

- `ScenarioStrategy` — ScamStrategy.ALL (all scam strategies).

#### `get_strategy_class() → type[ScenarioStrategy]`

Get the strategy enum class for this scenario.

**Returns:**

- `type[ScenarioStrategy]` — Type[ScenarioStrategy]: The ScamStrategy enum class.

#### `required_datasets() → list[str]`

Return a list of dataset names required by this scenario.

## `class ScamStrategy(ScenarioStrategy)`

Strategies for the Scam Scenario.

Non-Aggregate Values:
- ContextCompliance: This single-turn attack attempts to bypass safety measures by rephrasing the objective into
    a more benign context.
    It uses an adversarial chat target to:
    1) rephrase the objective (first user turn)
    2) generate the assistant's response to the benign question (first assistant turn)
    3) rephrase the original objective as a follow-up question (end of first assistant turn)
    This conversation is prepended and sent with an affirmative "yes" to get a response from the target.
- RolePlay: This single-turn attack uses the `persuasion_script_written.yaml` role-play scenario to convince the
    target to help draft a response to the scam objective. It is framed in the context of creating written samples
    to be used during training seminars.
- PersuasiveRedTeamingAttack: This multi-turn attack uses a persuasive persona with the `RedTeamingAttack` to
    iteratively convince the target to comply with the scam objective over multiple turns.

**Methods:**

#### `get_aggregate_tags() → set[str]`

Get the set of tags that represent aggregate categories.

**Returns:**

- `set[str]` — set[str]: Set of tags that are aggregate markers.
