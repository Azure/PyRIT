# pyrit.scenario.printer

Printer components for scenarios.

## `class ConsoleScenarioResultPrinter(ScenarioResultPrinter)`

Console printer for scenario results with enhanced formatting.

This printer formats scenario results for console display with optional color coding,
proper indentation, and visual separators. Colors can be disabled for consoles
that don't support ANSI characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Maximum width for text wrapping. Must be positive. Defaults to 100. Defaults to `100`. |
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. When False, all output will be plain text without colors. Defaults to True. Defaults to `True`. |
| `scorer_printer` | `Optional[ScorerPrinter]` | Printer for scorer information. If not provided, a ConsoleScorerPrinter with matching settings is created. Defaults to `None`. |

**Methods:**

#### `print_summary_async(result: ScenarioResult) → None`

Print a summary of the scenario result with per-strategy breakdown.

Displays:
- Scenario identification (name, version, PyRIT version)
- Target and scorer information
- Overall statistics
- Per-strategy success rates and result counts

| Parameter | Type | Description |
|---|---|---|
| `result` | `ScenarioResult` | The scenario result to summarize |

## `class ScenarioResultPrinter(ABC)`

Abstract base class for printing scenario results.

This interface defines the contract for printing scenario results in various formats.
Implementations can render results to console, logs, files, or other outputs.

**Methods:**

#### `print_summary_async(result: ScenarioResult) → None`

Print a summary of the scenario result with per-strategy breakdown.

Displays:
- Scenario identification (name, version, PyRIT version)
- Target information
- Overall statistics
- Per-strategy success rates and result counts

| Parameter | Type | Description |
|---|---|---|
| `result` | `ScenarioResult` | The scenario result to summarize |
