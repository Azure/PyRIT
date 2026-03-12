# pyrit.executor.core

Core executor module.

## `class Strategy(ABC, Generic[StrategyContextT, StrategyResultT])`

Abstract base class for strategies with enforced lifecycle management.

Ensures a consistent execution flow: validate -> setup -> execute -> teardown.
The teardown phase is guaranteed to run even if exceptions occur.

Subclasses must implement:
_validate_context(): Validate context
_setup_async(): Initialize resources
_perform_async(): Execute the logic
_teardown_async(): Clean up resources

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `context_type` | `type[StrategyContextT]` | The type of context this strategy will use. |
| `event_handler` | `Optional[StrategyEventHandler[StrategyContextT, StrategyResultT]]` | An optional event handler for strategy events. Defaults to `None`. |
| `logger` | `logging.Logger` | The logger to use for this strategy. Defaults to `logger`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → StrategyResultT`

Execute the strategy asynchronously with the given keyword arguments.

**Returns:**

- `StrategyResultT` — The result of the strategy execution.

#### `execute_with_context_async(context: StrategyContextT) → StrategyResultT`

Execute strategy with complete lifecycle management.

Enforces: validate -> setup -> execute -> teardown.

| Parameter | Type | Description |
|---|---|---|
| `context` | `StrategyContextT` | The context for the strategy, containing configuration and state. |

**Returns:**

- `StrategyResultT` — The result of the strategy execution, including outcome and reason.

**Raises:**

- `ValueError` — If the context validation fails.
- `RuntimeError` — If the strategy execution fails.

## `class StrategyContext(ABC)`

Base class for all strategy contexts.

**Methods:**

#### `duplicate() → StrategyContextT`

Create a deep copy of the context.

**Returns:**

- `StrategyContextT` — A deep copy of the context.

## `class StrategyConverterConfig`

Configuration for prompt converters used in strategies.

This class defines the converter configurations that transform prompts
during the strategy process, both for requests and responses.

## `class StrategyEvent(Enum)`

Enumeration of all strategy lifecycle events.

## `class StrategyEventData(Generic[StrategyContextT, StrategyResultT])`

Data passed to event observers.

## `class StrategyEventHandler(ABC, Generic[StrategyContextT, StrategyResultT])`

Abstract base class for strategy event handlers.

**Methods:**

#### on_event

```python
on_event(event_data: StrategyEventData[StrategyContextT, StrategyResultT]) → None
```

Handle a strategy event.

| Parameter | Type | Description |
|---|---|---|
| `event_data` | `StrategyEventData[StrategyContextT, StrategyResultT]` | Data about the event that occurred. |
