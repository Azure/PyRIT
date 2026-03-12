# pyrit.executor.promptgen.core

Core prompt generator strategy imports.

## `class PromptGeneratorStrategy(Strategy[PromptGeneratorStrategyContextT, PromptGeneratorStrategyResultT], ABC)`

Base class for all prompt generator strategies.
Provides a structure for implementing specific prompt generation strategies.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `context_type` | `type` | Type of the context used by the strategy. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |
| `event_handler` | `StrategyEventHandler` | Event handler for handling strategy events. Defaults to `None`. |

## `class PromptGeneratorStrategyContext(StrategyContext, ABC)`

Base class for all prompt generator strategy contexts.

## `class PromptGeneratorStrategyResult(StrategyResult, ABC)`

Base class for all prompt generator strategy results.
