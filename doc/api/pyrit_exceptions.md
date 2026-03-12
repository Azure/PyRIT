# pyrit.exceptions

Exception classes, retry helpers, and execution context utilities.

## Functions

### `clear_execution_context() → None`

Clear the current execution context.

### execution_context

```python
execution_context(component_role: ComponentRole, attack_strategy_name: Optional[str] = None, attack_identifier: Optional[ComponentIdentifier] = None, component_identifier: Optional[ComponentIdentifier] = None, objective_target_conversation_id: Optional[str] = None, objective: Optional[str] = None) → ExecutionContextManager
```

Create an execution context manager with the specified parameters.

| Parameter | Type | Description |
|---|---|---|
| `component_role` | `ComponentRole` | The role of the component being executed. |
| `attack_strategy_name` | `Optional[str]` | The name of the attack strategy class. Defaults to `None`. |
| `attack_identifier` | `Optional[ComponentIdentifier]` | The attack identifier. Defaults to `None`. |
| `component_identifier` | `Optional[ComponentIdentifier]` | The identifier from component.get_identifier(). Defaults to `None`. |
| `objective_target_conversation_id` | `Optional[str]` | The objective target conversation ID if available. Defaults to `None`. |
| `objective` | `Optional[str]` | The attack objective if available. Defaults to `None`. |

**Returns:**

- `ExecutionContextManager` — A context manager that sets/clears the context.

### `get_execution_context() → Optional[ExecutionContext]`

Get the current execution context.

**Returns:**

- `Optional[ExecutionContext]` — Optional[ExecutionContext]: The current context, or None if not set.

### `get_retry_max_num_attempts() → int`

Get the maximum number of retry attempts.

**Returns:**

- `int` — Maximum retry attempts.

### handle_bad_request_exception

```python
handle_bad_request_exception(response_text: str, request: MessagePiece, is_content_filter: bool = False, error_code: int = 400) → Message
```

Handle bad request responses and map them to standardized error messages.

| Parameter | Type | Description |
|---|---|---|
| `response_text` | `str` | Raw response text from the target. |
| `request` | `MessagePiece` | Original request piece that caused the error. |
| `is_content_filter` | `bool` | Whether the response is known to be content-filtered. Defaults to `False`. |
| `error_code` | `int` | Status code to include in the generated error payload. Defaults to `400`. |

**Returns:**

- `Message` — A constructed error response message.

**Raises:**

- `RuntimeError` — If the response does not match bad-request content-filter conditions.

### pyrit_custom_result_retry

```python
pyrit_custom_result_retry(retry_function: Callable[..., bool], retry_max_num_attempts: Optional[int] = None) → Callable[..., Any]
```

Apply retry logic with exponential backoff to a function.

Retries the function if the result of the retry_function is True,
with a wait time between retries that follows an exponential backoff strategy.
Logs retry attempts at the INFO level and stops after a maximum number of attempts.

| Parameter | Type | Description |
|---|---|---|
| `retry_function` | `Callable` | The boolean function to determine if a retry should occur based on the result of the decorated function. |
| `retry_max_num_attempts` | `(Optional, int)` | The maximum number of retry attempts. Defaults to environment variable CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS or 10. Defaults to `None`. |

**Returns:**

- `Callable[..., Any]` — The decorated function with retry logic applied.

### `pyrit_json_retry(func: Callable[..., Any]) → Callable[..., Any]`

Apply retry logic to a function.

Retries the function if it raises a JSON error.
Logs retry attempts at the INFO level and stops after a maximum number of attempts.

| Parameter | Type | Description |
|---|---|---|
| `func` | `Callable` | The function to be decorated. |

**Returns:**

- `Callable[..., Any]` — The decorated function with retry logic applied.

### `pyrit_placeholder_retry(func: Callable[..., Any]) → Callable[..., Any]`

Apply retry logic.

Retries the function if it raises MissingPromptPlaceholderException.
Logs retry attempts at the INFO level and stops after a maximum number of attempts.

| Parameter | Type | Description |
|---|---|---|
| `func` | `Callable` | The function to be decorated. |

**Returns:**

- `Callable[..., Any]` — The decorated function with retry logic applied.

### `pyrit_target_retry(func: Callable[..., Any]) → Callable[..., Any]`

Apply retry logic with exponential backoff to a function.

Retries the function if it raises RateLimitError or EmptyResponseException,
with a wait time between retries that follows an exponential backoff strategy.
Logs retry attempts at the INFO level and stops after a maximum number of attempts.

| Parameter | Type | Description |
|---|---|---|
| `func` | `Callable` | The function to be decorated. |

**Returns:**

- `Callable[..., Any]` — The decorated function with retry logic applied.

### `remove_markdown_json(response_msg: str) → str`

Remove markdown wrappers and return a JSON payload when possible.

| Parameter | Type | Description |
|---|---|---|
| `response_msg` | `str` | The response message to check. |

**Returns:**

- `str` — The response message without Markdown formatting if present.

### `set_execution_context(context: ExecutionContext) → None`

Set the current execution context.

| Parameter | Type | Description |
|---|---|---|
| `context` | `ExecutionContext` | The execution context to set. |

## `class BadRequestException(PyritException)`

Exception class for bad client requests.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `status_code` | `int` | Status code for the error. Defaults to `400`. |
| `message` | `str` | Error message. Defaults to `'Bad Request'`. |

## `class ComponentRole(Enum)`

Identifies the role of a component within an attack execution.

This enum is used to provide meaningful context in error messages and retry logs,
helping users identify which part of an attack encountered an issue.

## `class EmptyResponseException(BadRequestException)`

Exception class for empty response errors.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `status_code` | `int` | Status code for the error. Defaults to `204`. |
| `message` | `str` | Error message. Defaults to `'No Content'`. |

## `class ExecutionContext`

Holds context information about the currently executing component.

This context is used to enrich error messages and retry logs with
information about which component failed and its configuration.

**Methods:**

#### `get_exception_details() → str`

Generate detailed exception context for error messages.

**Returns:**

- `str` — A multi-line formatted string with full context details.

#### `get_retry_context_string() → str`

Generate a concise context string for retry log messages.

**Returns:**

- `str` — A formatted string with component role, component name, and endpoint.

## `class ExecutionContextManager`

A context manager for setting execution context during component operations.

This class provides a convenient way to set and automatically clear
execution context when entering and exiting a code block.

On successful exit, the context is restored to its previous value.
On exception, the context is preserved so exception handlers can access it.

## `class InvalidJsonException(PyritException)`

Exception class for blocked content errors.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | Error message. Defaults to `'Invalid JSON Response'`. |

## `class MissingPromptPlaceholderException(PyritException)`

Exception class for missing prompt placeholder errors.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | Error message. Defaults to `'No prompt placeholder'`. |

## `class PyritException(Exception, ABC)`

Base exception class for PyRIT components.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `status_code` | `int` | HTTP-style status code associated with the error. Defaults to `500`. |
| `message` | `str` | Human-readable error description. Defaults to `'An error occurred'`. |

**Methods:**

#### `process_exception() → str`

Log and return a JSON string representation of the exception.

**Returns:**

- `str` — Serialized status code and message.

## `class RateLimitException(PyritException)`

Exception class for authentication errors.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `status_code` | `int` | Status code for the error. Defaults to `429`. |
| `message` | `str` | Error message. Defaults to `'Rate Limit Exception'`. |
