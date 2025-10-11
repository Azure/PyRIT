# PyRIT Default Values System

This system provides a clean, decorator-based approach to managing default values for `__init__` parameters across class hierarchies.

## Overview

The `@apply_defaults` decorator automatically applies configured default values to `__init__` parameters, with support for inheritance hierarchies where more specific class defaults take precedence over parent class defaults.

## Key Components

### 1. `DefaultValueScope`
A frozen dataclass that defines the scope for a default value:
- `parameter_name`: The name of the parameter
- `class_type`: The class type for which the default applies

### 2. `@apply_defaults` Decorator
Decorates `__init__` methods to automatically apply default values from the global registry.

### 3. Helper Functions
- `set_default_value()`: Set a default value for a specific class and parameter
- `get_global_default_values()`: Access the global defaults registry

## Usage

### Basic Usage

```python
from pyrit.setup.pyrit_default_value import apply_defaults, set_default_value

class OpenAIChatTarget:
    @apply_defaults
    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self._temperature = temperature
        self._top_p = top_p

# Set defaults
set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
set_default_value(class_type=OpenAIChatTarget, parameter_name="top_p", value=0.9)

# Use defaults
target = OpenAIChatTarget()  # temperature=0.7, top_p=0.9

# Override defaults
target = OpenAIChatTarget(temperature=0.5)  # temperature=0.5, top_p=0.9
```

### Inheritance

The system respects class hierarchies. More specific (derived) class defaults take precedence:

```python
class AzureOpenAIChatTarget(OpenAIChatTarget):
    @apply_defaults
    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        api_version: Optional[str] = None,
    ) -> None:
        super().__init__(temperature=temperature, top_p=top_p)
        self._api_version = api_version

# Set parent defaults
set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
set_default_value(class_type=OpenAIChatTarget, parameter_name="top_p", value=0.9)

# Set subclass-specific defaults (overrides parent)
set_default_value(class_type=AzureOpenAIChatTarget, parameter_name="temperature", value=0.3)
set_default_value(class_type=AzureOpenAIChatTarget, parameter_name="api_version", value="2024-10-21")

# Result: AzureOpenAIChatTarget uses temperature=0.3 (more specific)
#         but inherits top_p=0.9 from parent
target = AzureOpenAIChatTarget()  # temperature=0.3, top_p=0.9, api_version="2024-10-21"
```

## Behavior

1. **Explicit values always win**: If a user provides a value, it's used regardless of configured defaults
2. **None is considered "not provided"**: Only when a parameter is None will defaults be applied
3. **Inheritance hierarchy**: Walks up the MRO (Method Resolution Order) to find the most specific default
4. **Non-intrusive**: Works seamlessly with existing code patterns

## Example Output

```
Example 1: No defaults configured
Initialized with temperature=None, top_p=None, max_tokens=None

Example 2: Set defaults for OpenAIChatTarget
Initialized with temperature=0.7, top_p=0.9, max_tokens=None

Example 3: Override with explicit values
Initialized with temperature=0.5, top_p=0.9, max_tokens=100

Example 4: Subclass inherits parent defaults
Initialized with temperature=0.7, top_p=0.9, max_tokens=None
Azure-specific: api_version=None

Example 5: Subclass-specific defaults
Initialized with temperature=0.3, top_p=0.9, max_tokens=None
Azure-specific: api_version=2024-10-21
```

## Implementation Details

- Uses Python's `inspect` module to analyze function signatures
- Leverages `functools.wraps` to preserve function metadata
- Frozen dataclass ensures `DefaultValueScope` can be used as dictionary key
- Global singleton pattern for centralized default management

## Benefits

1. **Clean API**: No need to pass class type or parameter name every time
2. **DRY Principle**: Set defaults once, apply everywhere
3. **Flexible**: Easy to override at instantiation time
4. **Inheritance-aware**: Respects class hierarchies automatically
5. **Type-safe**: Works with type hints and IDE autocompletion
