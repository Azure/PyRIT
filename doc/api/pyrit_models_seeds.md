# pyrit.models.seeds

Seeds module - Contains all seed-related classes for PyRIT.

This module provides the core seed types used throughout PyRIT:
- Seed: Base class for all seed types
- SeedPrompt: Seed with role and sequence for conversations
- SeedObjective: Seed representing an attack objective
- SeedGroup: Base container for grouping seeds
- SeedAttackGroup: Attack-specific seed group with objectives and prepended conversations
- SeedAttackTechniqueGroup: Technique-specific seed group where all seeds must be general strategies
- SeedSimulatedConversation: Configuration for generating simulated conversations
- SeedDataset: Container for managing collections of seeds

## `class NextMessageSystemPromptPaths(enum.Enum)`

Enum for predefined next message generation system prompt paths.

## `class Seed(YamlLoadable)`

Represents seed data with various attributes and metadata.

**Methods:**

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → Seed
```

Load a Seed from a YAML file and validate that it contains specific parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `Seed` — The loaded and validated seed of the specific subclass type.

#### `render_template_value(kwargs: Any = {}) → str`

Render self.value as a template with provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in the SeedPrompt value. Defaults to `{}`. |

**Returns:**

- `str` — A new prompt with the parameters applied.

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

#### `render_template_value_silent(kwargs: Any = {}) → str`

Render self.value as a template with provided parameters. For parameters in the template
that are not provided as kwargs here, this function will leave them as is instead of raising an error.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in the SeedPrompt value. Defaults to `{}`. |

**Returns:**

- `str` — A new prompt with the parameters applied.

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

#### `set_sha256_value_async() → None`

Compute the SHA256 hash value asynchronously.
It should be called after prompt `value` is serialized to text,
as file paths used in the `value` may have changed from local to memory storage paths.

Note, this method is async due to the blob retrieval. And because of that, we opted
to take it out of main and setter functions. The disadvantage is that it must be explicitly called.

## `class SeedAttackGroup(SeedGroup)`

A group of seeds for use in attack scenarios.

This class extends SeedGroup with attack-specific validation:
- Requires exactly one SeedObjective (not optional like in SeedGroup)

All other functionality (simulated conversation, prepended conversation,
next_message, etc.) is inherited from SeedGroup.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. Must include exactly one SeedObjective. |

**Methods:**

#### `validate() → None`

Validate the seed attack group state.

Extends SeedGroup validation to require exactly one objective.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedAttackTechniqueGroup(SeedGroup)`

A group of seeds representing a general attack technique.

This class extends SeedGroup with technique-specific validation:
- Requires all seeds to have is_general_technique=True

All other functionality (simulated conversation, prepended conversation,
next_message, etc.) is inherited from SeedGroup.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. All seeds must have is_general_technique=True. |

**Methods:**

#### `validate() → None`

Validate the seed attack technique group state.

Extends SeedGroup validation to require all seeds to be general strategies.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedDataset(YamlLoadable)`

SeedDataset manages seed prompts plus optional top-level defaults.
Prompts are stored as a Sequence[Seed], so references to prompt properties
are straightforward (e.g. ds.seeds[0].value).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Optional[Union[Sequence[dict[str, Any]], Sequence[Seed]]]` | List of seed dictionaries or Seed objects. Defaults to `None`. |
| `data_type` | `Optional[PromptDataType]` | Default data type for seeds. Defaults to `'text'`. |
| `name` | `Optional[str]` | Name of the dataset. Defaults to `None`. |
| `dataset_name` | `Optional[str]` | Dataset name for categorization. Defaults to `None`. |
| `harm_categories` | `Optional[Sequence[str]]` | List of harm categories. Defaults to `None`. |
| `description` | `Optional[str]` | Description of the dataset. Defaults to `None`. |
| `authors` | `Optional[Sequence[str]]` | List of authors. Defaults to `None`. |
| `groups` | `Optional[Sequence[str]]` | List of groups. Defaults to `None`. |
| `source` | `Optional[str]` | Source of the dataset. Defaults to `None`. |
| `date_added` | `Optional[datetime]` | Date when the dataset was added. Defaults to `None`. |
| `added_by` | `Optional[str]` | User who added the dataset. Defaults to `None`. |
| `seed_type` | `Optional[SeedType]` | The type of seeds in this dataset ("prompt", "objective", or "simulated_conversation"). Defaults to `None`. |
| `is_objective` | `bool` | Deprecated in 0.13.0. Use seed_type="objective" instead. Defaults to `False`. |

**Methods:**

#### `from_dict(data: dict[str, Any]) → SeedDataset`

Build a SeedDataset by merging top-level defaults into each item in `seeds`.

| Parameter | Type | Description |
|---|---|---|
| `data` | `Dict[str, Any]` | Dataset payload with top-level defaults and seed entries. |

**Returns:**

- `SeedDataset` — Constructed dataset with merged defaults.

**Raises:**

- `ValueError` — If any seed entry includes a pre-set prompt_group_id.

#### get_random_values

```python
get_random_values(number: PositiveInt, harm_categories: Optional[Sequence[str]] = None) → Sequence[str]
```

Extract and return random prompt values from the dataset.

| Parameter | Type | Description |
|---|---|---|
| `number` | `int` | The number of random prompt values to return. |
| `harm_categories` | `Optional[Sequence[str]]` | If provided, only prompts containing at least one of these harm categories are included. Defaults to `None`. |

**Returns:**

- `Sequence[str]` — Sequence[str]: A list of prompt values.

#### get_values

```python
get_values(first: Optional[PositiveInt] = None, last: Optional[PositiveInt] = None, harm_categories: Optional[Sequence[str]] = None) → Sequence[str]
```

Extract and return prompt values from the dataset.

| Parameter | Type | Description |
|---|---|---|
| `first` | `Optional[int]` | If provided, values from the first N prompts are included. Defaults to `None`. |
| `last` | `Optional[int]` | If provided, values from the last N prompts are included. Defaults to `None`. |
| `harm_categories` | `Optional[Sequence[str]]` | If provided, only prompts containing at least one of these harm categories are included. Defaults to `None`. |

**Returns:**

- `Sequence[str]` — Sequence[str]: A list of prompt values.

#### group_seed_prompts_by_prompt_group_id

```python
group_seed_prompts_by_prompt_group_id(seeds: Sequence[Seed]) → Sequence[SeedGroup]
```

Group the given list of seeds by prompt_group_id and create
SeedGroup or SeedAttackGroup instances.

For each group, this method first attempts to create a SeedAttackGroup
(which has attack-specific properties like objective). If validation fails,
it falls back to a basic SeedGroup.

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Seed]` | A list of Seed objects. |

**Returns:**

- `Sequence[SeedGroup]` — A list of SeedGroup or SeedAttackGroup objects, with seeds grouped by
- `Sequence[SeedGroup]` — prompt_group_id. Each group will be ordered by the sequence number of
- `Sequence[SeedGroup]` — the seeds, if available.

#### `render_template_value(kwargs: object = {}) → None`

Render seed values as templates using provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `object` | Key-value pairs to replace in the SeedDataset value. Defaults to `{}`. |

**Raises:**

- `ValueError` — If parameters are missing or invalid in the template.

## `class SeedGroup(YamlLoadable)`

A container for grouping prompts that need to be sent together.

This class handles:
- Grouping of SeedPrompt, SeedObjective, and SeedSimulatedConversation
- Consistent group IDs and roles across seeds
- Prepended conversation and next message extraction
- Validation of sequence overlaps between SeedPrompts and SeedSimulatedConversation

All prompts in the group share the same `prompt_group_id`.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Sequence[Union[Seed, dict[str, Any]]]` | Sequence of seeds. Can include: - SeedObjective (or dict with seed_type="objective") - SeedSimulatedConversation (or dict with seed_type="simulated_conversation") - SeedPrompt for prompts (or dict with seed_type="prompt" or no seed_type) Note: is_objective and is_simulated_conversation are deprecated since 0.13.0. |

**Methods:**

#### `is_single_part_single_text_request() → bool`

Check if this is a single text prompt.

**Returns:**

- `bool` — True when there is exactly one prompt and it is text.

#### `is_single_request() → bool`

Check if all prompts are in a single sequence.

**Returns:**

- `bool` — True when all prompts share one sequence number.

#### `is_single_turn() → bool`

Check if this is a single-turn group (single request without objective).

**Returns:**

- `bool` — True when the group is a single request and has no objective.

#### `render_template_value(kwargs: Any = {}) → None`

Render seed values as templates with provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Any` | Key-value pairs to replace in seed values. Defaults to `{}`. |

#### `validate() → None`

Validate the seed group state.

This method can be called after external modifications to seeds
to ensure the group remains in a valid state. It is automatically
called during initialization.

**Raises:**

- `ValueError` — If validation fails.

## `class SeedObjective(Seed)`

Represents a seed objective with various attributes and metadata.

**Methods:**

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedObjective
```

Load a Seed from a YAML file. Because SeedObjectives do not have any parameters, the required_parameters
and error_message arguments are unused.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `SeedObjective` — The loaded and validated seed of the specific subclass type.

## `class SeedPrompt(Seed)`

Represents a seed prompt with various attributes and metadata.

**Methods:**

#### from_messages

```python
from_messages(messages: list[Message], starting_sequence: int = 0, prompt_group_id: Optional[uuid.UUID] = None) → list[SeedPrompt]
```

Convert a list of Messages to a list of SeedPrompts.

Each MessagePiece becomes a SeedPrompt. All pieces from the same message
share the same sequence number, preserving the grouping.

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | List of Messages to convert. |
| `starting_sequence` | `int` | The starting sequence number. Defaults to 0. Defaults to `0`. |
| `prompt_group_id` | `Optional[uuid.UUID]` | Optional group ID to assign to all prompts. Defaults to None. Defaults to `None`. |

**Returns:**

- `list[SeedPrompt]` — List of SeedPrompts with incrementing sequence numbers per message.

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedPrompt
```

Load a Seed from a YAML file and validate that it contains specific parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the template. |
| `required_parameters` | `list[str]` | List of parameter names that must exist in the template. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. If None, a default message is used. Defaults to `None`. |

**Returns:**

- `SeedPrompt` — The loaded and validated SeedPrompt of the specific subclass type.

**Raises:**

- `ValueError` — If the template doesn't contain all required parameters.

#### `set_encoding_metadata() → None`

Set encoding metadata for the prompt within metadata dictionary. For images, this is just the
file format. For audio and video, this also includes bitrate (kBits/s as int), samplerate (samples/second
as int), bitdepth (as int), filesize (bytes as int), and duration (seconds as int) if the file type is
supported by TinyTag. Example supported file types include: MP3, MP4, M4A, and WAV.

## `class SeedSimulatedConversation(Seed)`

Configuration for generating a simulated conversation dynamically.

This class holds the paths and parameters needed to generate prepended conversation
content by running an adversarial chat against a simulated (compliant) target.

This is a pure configuration class. The actual generation is performed by
`generate_simulated_conversation_async` in the executor layer, which accepts
this config along with runtime dependencies (adversarial_chat target, scorer).

The `value` property returns a JSON serialization of the config for database
storage and deduplication.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `adversarial_chat_system_prompt_path` | `Union[str, Path]` | Path to YAML file containing the adversarial chat system prompt. |
| `simulated_target_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to YAML file containing the simulated target system prompt. Defaults to the compliant prompt. Defaults to `None`. |
| `next_message_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to YAML file containing the system prompt for generating a final user message. If provided, after the simulated conversation is generated, a single LLM call generates a user message that attempts to get the target to fulfill the objective. Defaults to None (no next message generation). Defaults to `None`. |
| `num_turns` | `int` | Number of conversation turns to generate. Defaults to 3. Defaults to `3`. |
| `sequence` | `int` | The starting sequence number for generated turns. When combined with static SeedPrompts, this determines where the simulated turns are inserted. Defaults to 0. Defaults to `0`. |
| `pyrit_version` | `Optional[str]` | PyRIT version for reproducibility tracking. Defaults to current version. Defaults to `None`. |
| `**kwargs` | `Any` | Additional arguments passed to the Seed base class. Defaults to `{}`. |

**Methods:**

#### `compute_hash() → str`

Compute a deterministic hash of this configuration.

**Returns:**

- `str` — A SHA256 hash string representing the configuration.

#### `from_dict(data: dict[str, Any]) → SeedSimulatedConversation`

Create a SeedSimulatedConversation from a dictionary, typically from YAML.

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict[str, Any]` | Dictionary containing the configuration. |

**Returns:**

- `SeedSimulatedConversation` — A new SeedSimulatedConversation instance.

**Raises:**

- `ValueError` — If required configuration fields are missing.

#### from_yaml_with_required_parameters

```python
from_yaml_with_required_parameters(template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None) → SeedSimulatedConversation
```

Load a SeedSimulatedConversation from a YAML file and validate required parameters.

| Parameter | Type | Description |
|---|---|---|
| `template_path` | `Union[str, Path]` | Path to the YAML file containing the config. |
| `required_parameters` | `list[str]` | List of parameter names that must exist. |
| `error_message` | `Optional[str]` | Custom error message if validation fails. Defaults to `None`. |

**Returns:**

- `SeedSimulatedConversation` — The loaded and validated SeedSimulatedConversation.

**Raises:**

- `ValueError` — If required parameters are missing.

#### `get_identifier() → dict[str, Any]`

Get an identifier dict capturing this configuration for comparison/storage.

**Returns:**

- `dict[str, Any]` — Dictionary with configuration details.

#### load_simulated_target_system_prompt

```python
load_simulated_target_system_prompt(objective: str, num_turns: int, simulated_target_system_prompt_path: Optional[Union[str, Path]] = None) → Optional[str]
```

Load and render the simulated target system prompt.

If no path is provided, returns None (no system prompt).
Validates that the template has required `objective` and `num_turns` parameters.

| Parameter | Type | Description |
|---|---|---|
| `objective` | `str` | The objective to render into the template. |
| `num_turns` | `int` | The number of turns to render into the template. |
| `simulated_target_system_prompt_path` | `Optional[Union[str, Path]]` | Optional path to the prompt YAML file. If None, no system prompt is used. Defaults to `None`. |

**Returns:**

- `Optional[str]` — The rendered system prompt string, or None if no path is provided.

**Raises:**

- `ValueError` — If the template doesn't have required parameters.

## `class SimulatedTargetSystemPromptPaths(enum.Enum)`

Enum for predefined simulated target system prompt paths.
