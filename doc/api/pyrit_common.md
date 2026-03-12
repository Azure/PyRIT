# pyrit.common

Common utilities and helpers for PyRIT.

## Functions

### `apply_defaults_to_method(method: Callable[..., T]) → Callable[..., T]`

Apply default values to a method's parameters.

This decorator looks up default values for the method's class and applies them
to parameters that are None or not provided.

| Parameter | Type | Description |
|---|---|---|
| `method` | `Callable[..., T]` | The method to decorate (typically __init__). |

**Returns:**

- `Callable[..., T]` — The decorated method.

### combine_dict

```python
combine_dict(existing_dict: Optional[dict[str, Any]] = None, new_dict: Optional[dict[str, Any]] = None) → dict[str, Any]
```

Combine two dictionaries containing string keys and values into one.

| Parameter | Type | Description |
|---|---|---|
| `existing_dict` | `Optional[dict[str, Any]]` | Dictionary with existing values Defaults to `None`. |
| `new_dict` | `Optional[dict[str, Any]]` | Dictionary with new values to be added to the existing dictionary. Note if there's a key clash, the value in new_dict will be used. Defaults to `None`. |

**Returns:**

- `dict[str, Any]` — combined dictionary

### combine_list

```python
combine_list(list1: Union[str, list[str]], list2: Union[str, list[str]]) → list[str]
```

Combine two lists or strings into a single list with unique values.

| Parameter | Type | Description |
|---|---|---|
| `list1` | `Union[str, List[str]]` | First list or string to combine. |
| `list2` | `Union[str, List[str]]` | Second list or string to combine. |

**Returns:**

- `list[str]` — Combined list containing unique values from both inputs.

### `convert_local_image_to_data_url(image_path: str) → str`

Convert a local image file to a data URL encoded in base64.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The file system path to the image file. |

**Returns:**

- `str` — A string containing the MIME type and the base64-encoded data of the image, formatted as a data URL.

**Raises:**

- `FileNotFoundError` — If no file is found at the specified `image_path`.
- `ValueError` — If the image file's extension is not in the supported formats list.

### `display_image_response(response_piece: MessagePiece) → None`

Display response images if running in notebook environment.

| Parameter | Type | Description |
|---|---|---|
| `response_piece` | `MessagePiece` | The response piece to display. |

### download_chunk

```python
download_chunk(url: str, headers: dict[str, str], start: int, end: int, client: httpx.AsyncClient) → bytes
```

Download a chunk of the file with a specified byte range.

**Returns:**

- `bytes` — The content of the downloaded chunk.

### `download_file(url: str, token: str, download_dir: Path, num_splits: int) → None`

Download a file in multiple segments (splits) using byte-range requests.

### download_files

```python
download_files(urls: list[str], token: str, download_dir: Path, num_splits: int = 3, parallel_downloads: int = 4) → None
```

Download multiple files with parallel downloads and segmented downloading.

### download_specific_files

```python
download_specific_files(model_id: str, file_patterns: list[str] | None, token: str, cache_dir: Path) → None
```

Download specific files from a Hugging Face model repository.
If file_patterns is None, downloads all files.

### `get_available_files(model_id: str, token: str) → list[str]`

Fetch available files for a model from the Hugging Face repository.

**Returns:**

- `list[str]` — List of available file names.

**Raises:**

- `ValueError` — If no files are found for the model.

### `get_global_default_values() → GlobalDefaultValues`

Get the global default values registry.

**Returns:**

- `GlobalDefaultValues` — The global default values registry instance.

### get_httpx_client

```python
get_httpx_client(use_async: bool = False, debug: bool = False, httpx_client_kwargs: Optional[Any] = {}) → httpx.Client | httpx.AsyncClient
```

Get the httpx client for making requests.

**Returns:**

- `httpx.Client | httpx.AsyncClient` — httpx.Client or httpx.AsyncClient: The configured httpx client.

### get_kwarg_param

```python
get_kwarg_param(kwargs: dict[str, Any], param_name: str, expected_type: type[_T], required: bool = True, default_value: Optional[_T] = None) → Optional[_T]
```

Validate and extract a parameter from kwargs.

| Parameter | Type | Description |
|---|---|---|
| `kwargs` | `Dict[str, Any]` | The dictionary containing parameters. |
| `param_name` | `str` | The name of the parameter to validate. |
| `expected_type` | `Type[_T]` | The expected type of the parameter. |
| `required` | `bool` | Whether the parameter is required. If True, raises ValueError if missing. Defaults to `True`. |
| `default_value` | `Optional[_T]` | Default value to return if the parameter is not required and not present. Defaults to `None`. |
| `kwargs` | `Dict[str, Any]` | The dictionary containing parameters. |
| `param_name` | `str` | The name of the parameter to validate. |
| `expected_type` | `Type[_T]` | The expected type of the parameter. |
| `required` | `bool` | Whether the parameter is required. If True, raises ValueError if missing. Defaults to `True`. |
| `default_value` | `Optional[_T]` | Default value to return if the parameter is not required and not present. Defaults to `None`. |

**Returns:**

- `Optional[_T]` — Optional[_T]: The validated parameter value if present and valid, otherwise None.
- `Optional[_T]` — Optional[_T]: The validated parameter value if present and valid, otherwise None.

**Raises:**

- `ValueError` — If the parameter is missing or None.
- `TypeError` — If the parameter is not of the expected type.

### get_non_required_value

```python
get_non_required_value(env_var_name: str, passed_value: Optional[str] = None) → str
```

Get a non-required value from an environment variable or a passed value,
preferring the passed value.

| Parameter | Type | Description |
|---|---|---|
| `env_var_name` | `str` | The name of the environment variable to check. |
| `passed_value` | `str` | The value passed to the function. Defaults to `None`. |

**Returns:**

- `str` — The passed value if provided, otherwise the value from the environment variable.
 If no value is found, returns an empty string.

### `get_random_indices(start: int, size: int, proportion: float) → list[int]`

Generate a list of random indices based on the specified proportion of a given size.
The indices are selected from the range [start, start + size).

| Parameter | Type | Description |
|---|---|---|
| `start` | `int` | Starting index (inclusive). It's the first index that could possibly be selected. |
| `size` | `int` | Size of the collection to select from. This is the total number of indices available. For example, if `start` is 0 and `size` is 10, the available indices are [0, 1, 2, ..., 9]. |
| `proportion` | `float` | The proportion of indices to select from the total size. Must be between 0 and 1. For example, if `proportion` is 0.5 and `size` is 10, 5 randomly selected indices will be returned. |

**Returns:**

- `list[int]` — List[int]: A list of randomly selected indices based on the specified proportion.

**Raises:**

- `ValueError` — If `start` is negative, `size` is not positive, or `proportion` is not between 0 and 1.

### `get_required_value(env_var_name: str, passed_value: Any) → Any`

Get a required value from an environment variable or a passed value,
preferring the passed value.

If no value is found, raises a KeyError

| Parameter | Type | Description |
|---|---|---|
| `env_var_name` | `str` | The name of the environment variable to check |
| `passed_value` | `Any` | The value passed to the function. Can be a string or a callable that returns a string. |

**Returns:**

- `Any` — The passed value if provided (preserving type for callables), otherwise the value from the environment variable.

**Raises:**

- `ValueError` — If neither the passed value nor the environment variable is provided.

### `is_in_ipython_session() → bool`

Determine if the code is running in an IPython session.

This may be useful if the behavior of the code should change when running in an IPython session.
For example, the code may display additional information or plots when running in an IPython session.

**Returns:**

- `bool` — True if the code is running in an IPython session, False otherwise.

### make_request_and_raise_if_error_async

```python
make_request_and_raise_if_error_async(endpoint_uri: str, method: str, post_type: PostType = 'json', debug: bool = False, extra_url_parameters: Optional[dict[str, str]] = None, request_body: Optional[dict[str, object]] = None, files: Optional[dict[str, tuple[str, bytes, str]]] = None, headers: Optional[dict[str, str]] = None, httpx_client_kwargs: Optional[Any] = {}) → httpx.Response
```

Make a request and raise an exception if it fails.

Query parameters can be specified either:
1. In the endpoint_uri (e.g., "https://api.com/endpoint?api-version=2024-10-21")
2. Via the extra_url_parameters dict
3. Both (extra_url_parameters will be merged with URL query parameters, with extra_url_parameters taking precedence)

**Returns:**

- `httpx.Response` — httpx.Response: The response from the request.

### print_deprecation_message

```python
print_deprecation_message(old_item: type | Callable[..., Any] | str, new_item: type | Callable[..., Any] | str, removed_in: str) → None
```

Emit a deprecation warning.

| Parameter | Type | Description |
|---|---|---|
| `old_item` | `type | Callable[..., Any] | str` | The deprecated class, function, or its string name |
| `new_item` | `type | Callable[..., Any] | str` | The replacement class, function, or its string name |
| `removed_in` | `str` | The version in which the deprecated item will be removed |

### `reset_default_values() → None`

Reset all default values in the global registry.

### set_default_value

```python
set_default_value(class_type: type[object], parameter_name: str, value: Any, include_subclasses: bool = True) → None
```

Set a default value for a specific class and parameter.

This is a convenience function that delegates to the global default values registry.

| Parameter | Type | Description |
|---|---|---|
| `class_type` | `type[object]` | The class type for which to set the default. |
| `parameter_name` | `str` | The name of the parameter to set the default for. |
| `value` | `Any` | The default value to set. |
| `include_subclasses` | `bool` | Whether this default should apply to subclasses as well. Defaults to `True`. |

### `verify_and_resolve_path(path: Union[str, Path]) → Path`

Verify that a path is valid and resolve it to an absolute path.

This utility function can be used anywhere path validation is needed,
such as in scorers, converters, or other components that accept file paths.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Union[str, Path]` | A path as a string or Path object. |

**Returns:**

- `Path` — The resolved absolute Path object.

**Raises:**

- `ValueError` — If the path is not a string or Path object.
- `FileNotFoundError` — If the path does not exist.

### warn_if_set

```python
warn_if_set(config: Any, unused_fields: list[str], log: Union[logging.Logger, logging.LoggerAdapter[logging.Logger]] = logger) → None
```

Warn about unused parameters in configurations.

This method checks if specified fields in a configuration object are set
(not None and not empty for collections) and logs a warning message for each
field that will be ignored by the current attack strategy.

| Parameter | Type | Description |
|---|---|---|
| `config` | `Any` | The configuration object to check for unused fields. |
| `unused_fields` | `List[str]` | List of field names to check in the config object. |
| `log` | `Union[logging.Logger, logging.LoggerAdapter]` | Logger to use for warning messages. Defaults to `logger`. |

## `class DefaultValueScope`

Represents a scope for default values with class type, parameter name, and inheritance rules.

This class defines the scope where a default value applies, including whether it should
be inherited by subclasses.

## `class Singleton(abc.ABCMeta)`

A metaclass for creating singleton classes. A singleton class can only have one instance.
If an instance of the class exists, it returns that instance; if not, it creates and returns a new one.

## `class YamlLoadable(abc.ABC)`

Abstract base class for objects that can be loaded from YAML files.

**Methods:**

#### `from_yaml_file(file: Union[Path | str]) → T`

Create a new object from a YAML file.

| Parameter | Type | Description |
|---|---|---|
| `file` | `Union[Path | str]` | The input file path. |

**Returns:**

- `T` — A new object of type T.

**Raises:**

- `FileNotFoundError` — If the input YAML file path does not exist.
- `ValueError` — If the YAML file is invalid.
