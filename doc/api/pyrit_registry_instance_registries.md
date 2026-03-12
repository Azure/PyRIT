# pyrit.registry.instance_registries

Instance registries package.

This package contains registries that store pre-configured instances (not classes).
Examples include ScorerRegistry which stores Scorer instances that have been
initialized with their required parameters (e.g., chat_target).

For registries that store classes (Type[T]), see class_registries/.

## `class BaseInstanceRegistry(ABC, RegistryProtocol[MetadataT], Generic[T, MetadataT])`

Abstract base class for registries that store pre-configured instances.

This class implements RegistryProtocol. Unlike BaseClassRegistry which stores
Type[T] and supports lazy discovery, instance registries store already-instantiated
objects that are registered explicitly (typically during initialization).

**Methods:**

#### `get(name: str) → Optional[T]`

Get a registered instance by name.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name of the instance. |

**Returns:**

- `Optional[T]` — The instance, or None if not found.

#### `get_all_instances() → dict[str, T]`

Get all registered instances as a name -> instance mapping.

**Returns:**

- `dict[str, T]` — Dict mapping registry names to their instances.

#### `get_names() → list[str]`

Get a sorted list of all registered names.

**Returns:**

- `list[str]` — Sorted list of registry names (keys).

#### `get_registry_singleton() → BaseInstanceRegistry[T, MetadataT]`

Get the singleton instance of this registry.

Creates the instance on first call with default parameters.

**Returns:**

- `BaseInstanceRegistry[T, MetadataT]` — The singleton instance of this registry class.

#### list_metadata

```python
list_metadata(include_filters: Optional[dict[str, object]] = None, exclude_filters: Optional[dict[str, object]] = None) → list[MetadataT]
```

List metadata for all registered instances, optionally filtered.

Supports filtering on any metadata property:
- Simple types (str, int, bool): exact match
- List types: checks if filter value is in the list

| Parameter | Type | Description |
|---|---|---|
| `include_filters` | `Optional[dict[str, object]]` | Optional dict of filters that items must match. Keys are metadata property names, values are the filter criteria. All filters must match (AND logic). Defaults to `None`. |
| `exclude_filters` | `Optional[dict[str, object]]` | Optional dict of filters that items must NOT match. Keys are metadata property names, values are the filter criteria. Any matching filter excludes the item. Defaults to `None`. |

**Returns:**

- `list[MetadataT]` — List of metadata dictionaries describing each registered instance.

#### `register(instance: T, name: str) → None`

Register an instance.

| Parameter | Type | Description |
|---|---|---|
| `instance` | `T` | The pre-configured instance to register. |
| `name` | `str` | The registry name for this instance. |

#### `reset_instance() → None`

Reset the singleton instance.

Useful for testing or reinitializing the registry.

## `class ConverterRegistry(BaseInstanceRegistry['PromptConverter', ComponentIdentifier])`

Registry for managing available converter instances.

This registry stores pre-configured PromptConverter instances (not classes).
Converters are registered explicitly via initializers after being instantiated
with their required parameters.

**Methods:**

#### `get_instance_by_name(name: str) → Optional[PromptConverter]`

Get a registered converter instance by name.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name of the converter. |

**Returns:**

- `Optional[PromptConverter]` — The converter instance, or None if not found.

#### `get_registry_singleton() → ConverterRegistry`

Get the singleton instance of the ConverterRegistry.

**Returns:**

- `ConverterRegistry` — The singleton ConverterRegistry instance.

#### `register_instance(converter: PromptConverter, name: Optional[str] = None) → None`

Register a converter instance.

| Parameter | Type | Description |
|---|---|---|
| `converter` | `PromptConverter` | The pre-configured converter instance (not a class). |
| `name` | `Optional[str]` | Optional custom registry name. If not provided, derived from the converter's unique identifier. Defaults to `None`. |

## `class ScorerRegistry(BaseInstanceRegistry['Scorer', ComponentIdentifier])`

Registry for managing available scorer instances.

This registry stores pre-configured Scorer instances (not classes).
Scorers are registered explicitly via initializers after being instantiated
with their required parameters (e.g., chat_target).

Scorers are identified by their snake_case name derived from the class name,
or a custom name provided during registration.

**Methods:**

#### `get_instance_by_name(name: str) → Optional[Scorer]`

Get a registered scorer instance by name.

Note: This returns an already-instantiated scorer, not a class.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name of the scorer. |

**Returns:**

- `Optional[Scorer]` — The scorer instance, or None if not found.

#### `get_registry_singleton() → ScorerRegistry`

Get the singleton instance of the ScorerRegistry.

**Returns:**

- `ScorerRegistry` — The singleton ScorerRegistry instance.

#### `register_instance(scorer: Scorer, name: Optional[str] = None) → None`

Register a scorer instance.

Note: Unlike ScenarioRegistry and InitializerRegistry which register classes,
ScorerRegistry registers pre-configured instances.

| Parameter | Type | Description |
|---|---|---|
| `scorer` | `Scorer` | The pre-configured scorer instance (not a class). |
| `name` | `Optional[str]` | Optional custom registry name. If not provided, derived from the scorer's unique identifier. Defaults to `None`. |

## `class TargetRegistry(BaseInstanceRegistry['PromptTarget', ComponentIdentifier])`

Registry for managing available prompt target instances.

This registry stores pre-configured PromptTarget instances (not classes).
Targets are registered explicitly via initializers after being instantiated
with their required parameters (e.g., endpoint, API keys).

Targets are identified by their snake_case name derived from the class name,
or a custom name provided during registration.

**Methods:**

#### `get_instance_by_name(name: str) → Optional[PromptTarget]`

Get a registered target instance by name.

Note: This returns an already-instantiated target, not a class.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name of the target. |

**Returns:**

- `Optional[PromptTarget]` — The target instance, or None if not found.

#### `get_registry_singleton() → TargetRegistry`

Get the singleton instance of the TargetRegistry.

**Returns:**

- `TargetRegistry` — The singleton TargetRegistry instance.

#### `register_instance(target: PromptTarget, name: Optional[str] = None) → None`

Register a target instance.

Note: Unlike ScenarioRegistry and InitializerRegistry which register classes,
TargetRegistry registers pre-configured instances.

| Parameter | Type | Description |
|---|---|---|
| `target` | `PromptTarget` | The pre-configured target instance (not a class). |
| `name` | `Optional[str]` | Optional custom registry name. If not provided, derived from class name with identifier hash appended (e.g., OpenAIChatTarget -> openai_chat_abc123). Defaults to `None`. |
