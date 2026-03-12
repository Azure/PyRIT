# pyrit.registry.class_registries

Class registries package.

This package contains registries that store classes (Type[T]) which can be
instantiated on demand. Examples include ScenarioRegistry and InitializerRegistry.

For registries that store pre-configured instances, see instance_registries/.

## `class BaseClassRegistry(ABC, RegistryProtocol[MetadataT], Generic[T, MetadataT])`

Abstract base class for registries that store classes (Type[T]).

This class implements RegistryProtocol and provides the common infrastructure
for class registries including:
- Lazy discovery of classes
- Registration of classes or factory callables
- Metadata caching
- Consistent API: get_class(), get_names(), list_metadata(), create_instance()
- Singleton pattern support via get_registry_singleton()

Subclasses must implement:
- _discover(): Populate the registry with discovered classes
- _build_metadata(): Build a metadata TypedDict for a class

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `lazy_discovery` | `bool` | If True, discovery is deferred until first access. If False, discovery runs immediately in constructor. Defaults to `True`. |

**Methods:**

#### `create_instance(name: str, kwargs: object = {}) → T`

Create an instance of a registered class.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name of the class. |
| `**kwargs` | `object` | Keyword arguments to pass to the factory or constructor. Defaults to `{}`. |

**Returns:**

- `T` — A new instance of type T.

**Raises:**

- `KeyError` — If the name is not registered.

#### `get_class(name: str) → type[T]`

Get a registered class by name.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name (snake_case identifier). |

**Returns:**

- `type[T]` — The registered class (Type[T]).
- `type[T]` — This returns the class itself, not an instance.

**Raises:**

- `KeyError` — If the name is not registered.

#### `get_entry(name: str) → Optional[ClassEntry[T]]`

Get the full ClassEntry for a registered class.

This is useful when you need access to factory or default_kwargs.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The registry name. |

**Returns:**

- `Optional[ClassEntry[T]]` — The ClassEntry containing class, factory, and defaults, or None if not found.

#### `get_names() → list[str]`

Get a sorted list of all registered names.

These are the snake_case registry keys (e.g., "encoding", "self_ask_refusal"),
not the actual class names (e.g., "EncodingScenario", "SelfAskRefusalScorer").

**Returns:**

- `list[str]` — Sorted list of registry names.

#### `get_registry_singleton() → BaseClassRegistry[T, MetadataT]`

Get the singleton instance of this registry.

Creates the instance on first call with default parameters.

**Returns:**

- `BaseClassRegistry[T, MetadataT]` — The singleton instance of this registry class.

#### list_metadata

```python
list_metadata(include_filters: Optional[dict[str, object]] = None, exclude_filters: Optional[dict[str, object]] = None) → list[MetadataT]
```

List metadata for all registered classes, optionally filtered.

Supports filtering on any metadata property:
- Simple types (str, int, bool): exact match
- List types: checks if filter value is in the list

| Parameter | Type | Description |
|---|---|---|
| `include_filters` | `Optional[dict[str, object]]` | Optional dict of filters that items must match. Keys are metadata property names, values are the filter criteria. All filters must match (AND logic). Defaults to `None`. |
| `exclude_filters` | `Optional[dict[str, object]]` | Optional dict of filters that items must NOT match. Keys are metadata property names, values are the filter criteria. Any matching filter excludes the item. Defaults to `None`. |

**Returns:**

- `list[MetadataT]` — List of metadata dictionaries (TypedDict) describing each registered class.
- `list[MetadataT]` — This returns descriptive info, not the classes themselves.

#### register

```python
register(name: Optional[str] = None, factory: Optional[Callable[..., T]] = None, default_kwargs: Optional[dict[str, object]] = None, description: Optional[str] = None) → None
```

Register a class with the registry.

| Parameter | Type | Description |
|---|---|---|
| `cls` | `type[T]` | The class to register (Type[T], not an instance). |
| `name` | `Optional[str]` | Optional custom registry name. If not provided, derived from class name. Defaults to `None`. |
| `factory` | `Optional[Callable[..., T]]` | Optional callable for creating instances with custom logic. Defaults to `None`. |
| `default_kwargs` | `Optional[dict[str, object]]` | Default keyword arguments for instance creation. Defaults to `None`. |
| `description` | `Optional[str]` | Optional description override. Defaults to `None`. |

#### `reset_instance() → None`

Reset the singleton instance.

Useful for testing or when re-discovery is needed.

## `class ClassEntry(Generic[T])`

Internal wrapper for a registered class.

This holds the class itself (Type[T]) along with optional factory
and default parameters for creating instances.

Note: This is an internal implementation detail. Users interact with
registries via get_class(), create_instance(), and list_metadata().

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `registered_class` | `type[T]` | The actual Python class (Type[T]). |
| `factory` | `Optional[Callable[..., T]]` | Optional callable that creates an instance. Defaults to `None`. |
| `default_kwargs` | `Optional[dict[str, object]]` | Default keyword arguments for instantiation. Defaults to `None`. |
| `description` | `Optional[str]` | Optional description override. Defaults to `None`. |

**Methods:**

#### `create_instance(kwargs: object = {}) → T`

Create an instance of the registered class.

| Parameter | Type | Description |
|---|---|---|
| `**kwargs` | `object` | Additional keyword arguments. These override default_kwargs. Defaults to `{}`. |

**Returns:**

- `T` — An instance of type T.

## `class InitializerMetadata(ClassRegistryEntry)`

Metadata describing a registered PyRITInitializer class.

Use get_class() to get the actual class.

## `class InitializerRegistry(BaseClassRegistry['PyRITInitializer', InitializerMetadata])`

Registry for discovering and managing available initializers.

This class discovers all PyRITInitializer subclasses from the
pyrit/setup/initializers directory structure.

Initializers are identified by their filename (e.g., "objective_target", "simple").
The directory structure is used for organization but not exposed to users.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `discovery_path` | `Optional[Path]` | The path to discover initializers from. If None, defaults to pyrit/setup/initializers (discovers all). To discover only scenarios, pass pyrit/setup/initializers/scenarios. Defaults to `None`. |
| `lazy_discovery` | `bool` | If True, discovery is deferred until first access. Defaults to False for backwards compatibility. Defaults to `False`. |

**Methods:**

#### `get_registry_singleton() → InitializerRegistry`

Get the singleton instance of the InitializerRegistry.

**Returns:**

- `InitializerRegistry` — The singleton InitializerRegistry instance.

#### `resolve_initializer_paths(initializer_names: list[str]) → list[Path]`

Resolve initializer names to their file paths.

| Parameter | Type | Description |
|---|---|---|
| `initializer_names` | `list[str]` | List of initializer names to resolve. |

**Returns:**

- `list[Path]` — List of resolved file paths.

**Raises:**

- `ValueError` — If any initializer name is not found or has no file path.

#### `resolve_script_paths(script_paths: list[str]) → list[Path]`

Resolve and validate custom script paths.

| Parameter | Type | Description |
|---|---|---|
| `script_paths` | `list[str]` | List of script path strings to resolve. |

**Returns:**

- `list[Path]` — List of resolved Path objects.

**Raises:**

- `FileNotFoundError` — If any script path does not exist.

## `class ScenarioMetadata(ClassRegistryEntry)`

Metadata describing a registered Scenario class.

Use get_class() to get the actual class.

## `class ScenarioRegistry(BaseClassRegistry['Scenario', ScenarioMetadata])`

Registry for discovering and managing available scenario classes.

This class discovers all Scenario subclasses from:
1. Built-in scenarios in pyrit.scenario.scenarios module
2. User-defined scenarios from initialization scripts (set via globals)

Scenarios are identified by their simple name (e.g., "encoding", "foundry").

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `lazy_discovery` | `bool` | If True, discovery is deferred until first access. Defaults to True for performance. Defaults to `True`. |

**Methods:**

#### `discover_user_scenarios() → None`

Discover user-defined scenarios from global variables.

After initialization scripts are executed, they may define Scenario subclasses
and store them in globals. This method searches for such classes.

User scenarios will override built-in scenarios with the same name.

#### `get_registry_singleton() → ScenarioRegistry`

Get the singleton instance of the ScenarioRegistry.

**Returns:**

- `ScenarioRegistry` — The singleton ScenarioRegistry instance.
