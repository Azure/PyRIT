# pyrit.setup

Module containing initialization PyRIT.

## Functions

### initialize_from_config_async

```python
initialize_from_config_async(config_path: Optional[Union[str, pathlib.Path]] = None) → ConfigurationLoader
```

Initialize PyRIT from a configuration file.

This is a convenience function that loads a ConfigurationLoader from
a YAML file and initializes PyRIT.

| Parameter | Type | Description |
|---|---|---|
| `config_path` | `Optional[Union[str, pathlib.Path]]` | Path to the configuration file. If None, uses the default path (~/.pyrit/.pyrit_conf). Can be a string or pathlib.Path. Defaults to `None`. |

**Returns:**

- `ConfigurationLoader` — The loaded ConfigurationLoader instance.

**Raises:**

- `FileNotFoundError` — If the configuration file does not exist.
- `ValueError` — If the configuration is invalid.

### initialize_pyrit_async

```python
initialize_pyrit_async(memory_db_type: Union[MemoryDatabaseType, str], initialization_scripts: Optional[Sequence[Union[str, pathlib.Path]]] = None, initializers: Optional[Sequence[PyRITInitializer]] = None, env_files: Optional[Sequence[pathlib.Path]] = None, silent: bool = False, memory_instance_kwargs: Any = {}) → None
```

Initialize PyRIT with the provided memory instance and loads environment files.

| Parameter | Type | Description |
|---|---|---|
| `memory_db_type` | `MemoryDatabaseType` | The MemoryDatabaseType string literal which indicates the memory instance to use for central memory. Options include "InMemory", "SQLite", and "AzureSQL". |
| `initialization_scripts` | `Optional[Sequence[Union[str, pathlib.Path]]]` | Optional sequence of Python script paths that contain PyRITInitializer classes. Each script must define either a get_initializers() function or an 'initializers' variable that returns/contains a list of PyRITInitializer instances. Defaults to `None`. |
| `initializers` | `Optional[Sequence[PyRITInitializer]]` | Optional sequence of PyRITInitializer instances to execute directly. These provide type-safe, validated configuration with clear documentation. Defaults to `None`. |
| `env_files` | `Optional[Sequence[pathlib.Path]]` | Optional sequence of environment file paths to load in order. If not provided, will load default .env and .env.local files from PyRIT home if they exist. All paths must be valid pathlib.Path objects. Defaults to `None`. |
| `silent` | `bool` | If True, suppresses print statements about environment file loading. Defaults to False. Defaults to `False`. |
| `**memory_instance_kwargs` | `Optional[Any]` | Additional keyword arguments to pass to the memory instance. Defaults to `{}`. |

**Raises:**

- `ValueError` — If an unsupported memory_db_type is provided or if env_files contains non-existent files.

## `class ConfigurationLoader(YamlLoadable)`

Loader for PyRIT configuration from YAML files.

This class loads configuration from a YAML file and provides methods to
initialize PyRIT with the loaded configuration.

**Methods:**

#### `from_dict(data: dict[str, Any]) → ConfigurationLoader`

Create a ConfigurationLoader from a dictionary.

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict[str, Any]` | Dictionary containing configuration values. |

**Returns:**

- `ConfigurationLoader` — A new ConfigurationLoader instance.

#### `get_default_config_path() → pathlib.Path`

Get the default configuration file path.

**Returns:**

- `pathlib.Path` — Path to the default config file in ~/.pyrit/.pyrit_conf

#### `initialize_pyrit_async() → None`

Initialize PyRIT with the loaded configuration.

This method resolves all initializer names to instances and calls
the core initialize_pyrit_async function.

**Raises:**

- `ValueError` — If configuration is invalid or initializers cannot be resolved.

#### load_with_overrides

```python
load_with_overrides(config_file: Optional[pathlib.Path] = None, memory_db_type: Optional[str] = None, initializers: Optional[Sequence[Union[str, dict[str, Any]]]] = None, initialization_scripts: Optional[Sequence[str]] = None, env_files: Optional[Sequence[str]] = None) → ConfigurationLoader
```

Load configuration with optional overrides.

This factory method implements a 3-layer configuration precedence:
1. Default config file (~/.pyrit/.pyrit_conf) if it exists
2. Explicit config_file argument if provided
3. Individual override arguments (non-None values take precedence)

This is a staticmethod (not classmethod) because it's a pure factory function
that doesn't need access to class state and can be reused by multiple interfaces
(CLI, shell, programmatic API).

| Parameter | Type | Description |
|---|---|---|
| `config_file` | `Optional[pathlib.Path]` | Optional path to a YAML-formatted configuration file. Defaults to `None`. |
| `memory_db_type` | `Optional[str]` | Override for database type (in_memory, sqlite, azure_sql). Defaults to `None`. |
| `initializers` | `Optional[Sequence[Union[str, dict[str, Any]]]]` | Override for initializer list. Defaults to `None`. |
| `initialization_scripts` | `Optional[Sequence[str]]` | Override for initialization script paths. Defaults to `None`. |
| `env_files` | `Optional[Sequence[str]]` | Override for environment file paths. Defaults to `None`. |

**Returns:**

- `ConfigurationLoader` — A merged ConfigurationLoader instance.

**Raises:**

- `FileNotFoundError` — If an explicitly specified config_file does not exist.
- `ValueError` — If the configuration is invalid.
