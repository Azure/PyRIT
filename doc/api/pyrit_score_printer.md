# pyrit.score.printer

Scorer printer classes for displaying scorer information in various formats.

## `class ConsoleScorerPrinter(ScorerPrinter)`

Console printer for scorer information with enhanced formatting.

This printer formats scorer details for console display with optional color coding,
proper indentation, and visual hierarchy. Colors can be disabled for consoles
that don't support ANSI characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `indent_size` | `int` | Number of spaces for indentation. Must be non-negative. Defaults to 2. Defaults to `2`. |
| `enable_colors` | `bool` | Whether to enable ANSI color output. When False, all output will be plain text without colors. Defaults to True. Defaults to `True`. |

**Methods:**

#### print_harm_scorer

```python
print_harm_scorer(scorer_identifier: ComponentIdentifier, harm_category: str) → None
```

Print harm scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Harm evaluation metrics (MAE, Krippendorff alpha) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |
| `harm_category` | `str` | The harm category for looking up metrics (e.g., "hate_speech", "violence"). |

#### `print_objective_scorer(scorer_identifier: ComponentIdentifier) → None`

Print objective scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Objective evaluation metrics (accuracy, precision, recall, F1) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |

## `class ScorerPrinter(ABC)`

Abstract base class for printing scorer information.

This interface defines the contract for printing scorer details including
type information, nested sub-scorers, and evaluation metrics from the registry.
Implementations can render output to console, logs, files, or other outputs.

**Methods:**

#### print_harm_scorer

```python
print_harm_scorer(scorer_identifier: ComponentIdentifier, harm_category: str) → None
```

Print harm scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Harm evaluation metrics (MAE, Krippendorff alpha) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |
| `harm_category` | `str` | The harm category for looking up metrics (e.g., "hate_speech", "violence"). |

#### `print_objective_scorer(scorer_identifier: ComponentIdentifier) → None`

Print objective scorer information including type, nested scorers, and evaluation metrics.

This method displays:
- Scorer type and identity information
- Nested sub-scorers (for composite scorers)
- Objective evaluation metrics (accuracy, precision, recall, F1) from the registry

| Parameter | Type | Description |
|---|---|---|
| `scorer_identifier` | `ComponentIdentifier` | The scorer identifier to print information for. |
